package provider

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"strings"
	"sync"
	"unicode"

	"github.com/hoveychen/mimic/pkg/client"
	"github.com/hoveychen/mimic/pkg/utils"
	"github.com/pkg/errors"
	"github.com/sirupsen/logrus"
	"github.com/sugarme/tokenizer"
	"github.com/sugarme/tokenizer/pretrained"
)

func init() {
	client.RegisterProvider("azure", NewAzureStub)
}

const AzureAPIVersion = "2023-05-15"

type AzureStub struct {
	tk                *tokenizer.Tokenizer
	censoredRegexp    *regexp.Regexp
	cjkCensoredRegexp *regexp.Regexp
	lock              sync.Mutex
	conf              *client.SubscriptionConfig
	concurrent        chan struct{}
}

func NewAzureStub(conf *client.SubscriptionConfig) client.Stub {
	configFile, err := tokenizer.CachedPath("gpt2", "tokenizer.json")
	utils.PanicError(err)

	tk, err := pretrained.FromFile(configFile)
	utils.PanicError(err)

	s := &AzureStub{
		tk:         tk,
		conf:       conf,
		concurrent: make(chan struct{}, conf.Concurrent),
	}
	s.buildCensoredRegexp()
	return s
}

func (s *AzureStub) MaxTokens(model client.ModelType) int {
	switch model {
	// chat completion
	case client.ModelGpt4:
		return 30000 // maximum 128000, too expensive to allow larger
	case client.ModelGpt4Vision:
		return 30000 // maximum 128000, too expensive to allow larger
	case client.ModelGpt3:
		return 15000 // maximum 16385
	// embeddings
	case client.ModelTextEmbeddingAda:
		return 8000
	// image generation
	case client.ModelDalle2, client.ModelDalle3, client.ModelDalle3HD:
		return 0
	// speech generation
	case client.ModelTTS1, client.ModelTTS1HD:
		return 0
	default:
		return 2000 // the worest LLM model has 2k context window
	}
}

func (s *AzureStub) buildCensoredRegexp() {
	var censoredWords []string
	var cjkCensoredWords []string

	for _, word := range s.conf.CensoredWords {
		if isCJK(word) {
			cjkCensoredWords = append(cjkCensoredWords, word)
		} else {
			censoredWords = append(censoredWords, word)
		}
	}
	if len(censoredWords) > 0 {
		s.censoredRegexp = regexp.MustCompile(`(?i)\b(` + strings.Join(censoredWords, "|") + `)\b`)
	}
	if len(cjkCensoredWords) > 0 {
		s.cjkCensoredRegexp = regexp.MustCompile(`(?i)(` + strings.Join(cjkCensoredWords, "|") + `)`)
	}
}

func isCJK(word string) bool {
	for _, r := range []rune(word) {
		if unicode.Is(unicode.Han, r) {
			return true
		}
	}
	return false
}

func (s *AzureStub) do(ctx context.Context, model client.ModelType, method, path string, data []byte) (*http.Response, error) {
	s.concurrent <- struct{}{}
	defer func() {
		<-s.concurrent
	}()

	deployment := s.getDeployment(model)
	req, err := http.NewRequestWithContext(ctx, method, fmt.Sprintf("%s/%s%s?api-version=%s", s.conf.Endpoint, deployment, path, AzureAPIVersion), bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	req.Header.Add("Content-Type", "application/json")
	req.Header.Add("api-key", s.conf.ApiSecret)

	return http.DefaultClient.Do(req)
}

func (s *AzureStub) Chat(ctx context.Context, model client.ModelType, conf *client.CallConfig, prompt []*client.OpenAIMessage) (*client.OpenAIMessage, *client.Usage, error) {
	if err := s.validateChatPrompt(model, prompt); err != nil {
		return nil, nil, err
	}

	body, err := json.Marshal(map[string]interface{}{
		"model":       s.getModelName(model),
		"messages":    prompt,
		"temperature": conf.Temperature,
	})
	if err != nil {
		return nil, nil, err
	}

	body = s.removeCensoredWords(body)
	resp, err := s.do(ctx, model, "POST", "/chat/completions", body)
	if err != nil {
		return nil, nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		if strings.Contains(string(body), "429") {
			return nil, nil, client.ErrTooManyRequests
		}
		return nil, nil, fmt.Errorf("unexpected status : %s %s", resp.Status, string(body))
	}

	var result struct {
		ID      string `json:"id"`
		Object  string `json:"object"`
		Model   string `json:"model"`
		Created int64  `json:"created"`
		Choices []struct {
			Message      *client.OpenAIMessage `json:"message"`
			Index        int                   `json:"index"`
			FinishReason string                `json:"finish_reason"`
		} `json:"choices"`
		Usage struct {
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
			TotalTokens      int `json:"total_tokens"`
		} `json:"usage"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, nil, err
	}
	if len(result.Choices) == 0 {
		return nil, nil, fmt.Errorf("no embedding")
	}

	for _, data := range result.Choices {
		usage := &client.Usage{
			PromptTokens:     result.Usage.PromptTokens,
			CompletionTokens: result.Usage.CompletionTokens,
			Model:            result.Model,
			Subscription:     s.conf.Name,
		}
		return data.Message, usage, nil
	}

	return nil, nil, fmt.Errorf("no result returned")
}

func (s *AzureStub) validateChatPrompt(model client.ModelType, prompt []*client.OpenAIMessage) error {
	for i, p := range prompt {
		switch p.GetRole() {
		case "system", "assistant", "user":
		default:
			return errors.Errorf("invalid role: %s", p.GetRole())
		}
		if p.GetContent() == "" {
			return errors.Errorf("empty prompt, idx: %d", i)
		}
	}

	// Check if the prompt is too long.
	var texts []string
	for _, p := range prompt {
		texts = append(texts, p.GetContent())
	}
	if s.MaxTokens(model) <= s.calcNumToken(strings.Join(texts, ";")) {
		return errors.New("prompt too long")
	}

	return nil
}

func (s *AzureStub) ChatStream(ctx context.Context, model client.ModelType, conf *client.CallConfig, prompt []*client.OpenAIMessage) (chan *client.OpenAIMessage, error) {
	if err := s.validateChatPrompt(model, prompt); err != nil {
		return nil, err
	}

	body, err := json.Marshal(map[string]interface{}{
		"model":       s.getModelName(model),
		"messages":    prompt,
		"temperature": conf.Temperature,
		"stream":      true,
	})
	if err != nil {
		return nil, err
	}

	body = s.removeCensoredWords(body)
	resp, err := s.do(ctx, model, "POST", "/chat/completions", body)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		if strings.Contains(string(body), "429") {
			return nil, client.ErrTooManyRequests
		}
		return nil, fmt.Errorf("unexpected status : %s %s", resp.Status, string(body))
	}

	ret := make(chan *client.OpenAIMessage, 1024)

	go func() {
		defer close(ret)
		defer resp.Body.Close()
		r := NewEventStreamReader(resp.Body, 4096)
		for {
			ev, err := r.ReadEvent()
			if err == io.EOF {
				return
			}
			if err != nil {
				logrus.Errorf("failed to read event: %v", err)
				return
			}

			if string(ev.Data) == "[DONE]" {
				return
			}

			var result struct {
				ID      string `json:"id"`
				Object  string `json:"object"`
				Model   string `json:"model"`
				Created int64  `json:"created"`
				Choices []struct {
					Delta        *client.OpenAIMessage `json:"delta"`
					Index        int                   `json:"index"`
					FinishReason string                `json:"finish_reason"`
				} `json:"choices"`
			}
			if err := json.NewDecoder(bytes.NewReader(ev.Data)).Decode(&result); err != nil {
				logrus.Errorf("failed to decode json: %v", err)
				return
			}
			if len(result.Choices) == 0 {
				logrus.Errorf("no result returned")
				return
			}

			if result.Choices[0].FinishReason == "stop" {
				return
			}

			ret <- result.Choices[0].Delta
		}
	}()

	return ret, nil
}

func (s *AzureStub) Embeddings(ctx context.Context, model client.ModelType, texts []string) ([][]float32, *client.Usage, error) {
	if err := s.validateEmbeddings(model, texts); err != nil {
		return nil, nil, err
	}

	var ret [][]float32
	usage := &client.Usage{
		Subscription: s.conf.Name,
	}
	for _, text := range texts {
		body, err := json.Marshal(map[string]interface{}{
			"input": text,
			"model": s.getModelName(model),
		})
		if err != nil {
			return nil, nil, err
		}
		body = s.removeCensoredWords(body)

		resp, err := s.do(ctx, model, "POST", "/embeddings", body)
		if err != nil {
			return nil, nil, err
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			body, _ := io.ReadAll(resp.Body)
			if strings.Contains(string(body), "429") {
				return nil, nil, client.ErrTooManyRequests
			}
			return nil, nil, fmt.Errorf("unexpected status : %s %s", resp.Status, string(body))
		}

		var result struct {
			Object string `json:"object"`
			Data   []struct {
				Embedding []float32 `json:"embedding"`
				Index     int       `json:"index"`
				Object    string    `json:"object"`
			} `json:"data"`
			Model string `json:"model"`
			Usage struct {
				PromptTokens int `json:"prompt_tokens"`
				TotalTokens  int `json:"total_tokens"`
			} `json:"usage"`
		}
		if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
			return nil, nil, err
		}
		if len(result.Data) == 0 {
			return nil, nil, fmt.Errorf("no embedding")
		}

		for _, data := range result.Data {
			ret = append(ret, data.Embedding)
		}

		usage.Model = result.Model
		usage.PromptTokens += result.Usage.PromptTokens
	}

	return ret, usage, nil
}

func (s *AzureStub) validateEmbeddings(model client.ModelType, texts []string) error {
	if len(texts) == 0 {
		return errors.New("empty text")
	}

	// Check if the texts is too long.
	for i, text := range texts {
		if text == "" {
			return errors.Errorf("empty text. index=%d", i)
		}
		if s.MaxTokens(model) <= s.calcNumToken(text) {
			return errors.Errorf("text too long. index=%d", i)
		}
	}
	return nil
}

// calcNumToken 函数根据给定的分词器和内容计算令牌数量
func (s *AzureStub) calcNumToken(content string) int {
	s.lock.Lock()
	defer s.lock.Unlock()
	ret, err := s.tk.EncodeSingle(content)
	if err != nil {
		logrus.Warnf("EncodeSingle: %s err=%v", content, err)
		return 0
	}
	return len(ret.GetTokens()) + 4
}

func (s *AzureStub) removeCensoredWords(data []byte) []byte {
	if s.censoredRegexp != nil {
		data = s.censoredRegexp.ReplaceAll(data, nil)
	}
	if s.cjkCensoredRegexp != nil {
		data = s.cjkCensoredRegexp.ReplaceAll(data, nil)
	}
	return data
}

func (s *AzureStub) getDeployment(model client.ModelType) string {
	switch model {
	case client.ModelGpt3:
		return "gpt-35"
	case client.ModelGpt4:
		return "gpt-4"
	case client.ModelGpt4_32k:
		return "gpt-4-32k"
	case client.ModelTextEmbeddingAda:
		return "ada-002"
	default:
		return ""
	}
}

func (s *AzureStub) getModelName(model client.ModelType) string {
	switch model {
	case client.ModelGpt3:
		return "gpt-35-turbo"
	case client.ModelGpt4:
		return "gpt-4"
	case client.ModelGpt4_32k:
		return "gpt-4-32k"
	case client.ModelTextEmbeddingAda:
		return "text-embedding-ada-002"
	default:
		return ""
	}
}
