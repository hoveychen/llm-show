package provider

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"

	"github.com/hoveychen/mimic/pkg/client"
	"github.com/hoveychen/mimic/pkg/utils"
	"github.com/pkg/errors"
	"github.com/samber/lo"
	"github.com/sirupsen/logrus"
	"github.com/sugarme/tokenizer"
	"github.com/sugarme/tokenizer/pretrained"
)

func init() {
	client.RegisterProvider("anthropic", NewAnthropicStub)
}

type AnthropicStub struct {
	tk         *tokenizer.Tokenizer
	lock       sync.Mutex
	conf       *client.SubscriptionConfig
	concurrent chan struct{}
}

func NewAnthropicStub(conf *client.SubscriptionConfig) client.Stub {
	configFile, err := tokenizer.CachedPath("gpt2", "tokenizer.json")
	utils.PanicError(err)

	tk, err := pretrained.FromFile(configFile)
	utils.PanicError(err)

	return &AnthropicStub{
		conf:       conf,
		tk:         tk,
		concurrent: make(chan struct{}, conf.Concurrent),
	}
}

func (s *AnthropicStub) MaxTokens(model client.ModelType) int {
	switch model {
	// chat completion
	case client.ModelClaude2, client.ModelClaudeInstant1:
		return 10000
	default:
		return 2000 // the worest LLM model has 2k context window
	}
}

func (s *AnthropicStub) do(ctx context.Context, method, path string, data []byte) (*http.Response, error) {
	s.concurrent <- struct{}{}
	defer func() {
		<-s.concurrent
	}()
	req, err := http.NewRequestWithContext(ctx, method, s.conf.Endpoint+path, bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	req.Header.Add("Content-Type", "application/json")
	req.Header.Add("Accept", "application/json")
	req.Header.Add("Anthropic-Version", "2023-06-01")
	req.Header.Add("x-api-key", s.conf.ApiSecret)

	return http.DefaultClient.Do(req)
}

func (s *AnthropicStub) getModelName(model client.ModelType) string {
	switch model {
	case client.ModelClaude2, client.ModelClaude2_100k:
		return "claude-2"
	case client.ModelClaudeInstant1, client.ModelClaudeInstant1_100k:
		return "claude-instant-1"
	default:
		return ""
	}
}

func (s *AnthropicStub) Chat(ctx context.Context, model client.ModelType, conf *client.CallConfig, prompt []*client.OpenAIMessage) (*client.OpenAIMessage, *client.Usage, error) {
	if err := s.validateChatPrompt(model, prompt); err != nil {
		return nil, nil, err
	}

	text, err := s.genPrompt(prompt)
	if err != nil {
		return nil, nil, err
	}

	body, err := json.Marshal(map[string]interface{}{
		"model":                s.getModelName(model),
		"prompt":               text,
		"max_tokens_to_sample": 4096,
		"temperature":          conf.Temperature,
	})
	if err != nil {
		return nil, nil, err
	}

	resp, err := s.do(ctx, "POST", "/v1/complete", body)
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
		Completion string `json:"completion"`
		StopReason string `json:"stop_reason"`
		Model      string `json:"model"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, nil, err
	}

	if result.StopReason == "max_tokens" {
		logrus.Warn("max tokens reached")
	}

	usage := &client.Usage{
		Model:        result.Model,
		Subscription: s.conf.Name,
	}

	return &client.OpenAIMessage{
		Role:    "assistant",
		Content: result.Completion,
	}, usage, nil
}

func (s *AnthropicStub) ChatStream(ctx context.Context, model client.ModelType, conf *client.CallConfig, prompt []*client.OpenAIMessage) (chan *client.OpenAIMessage, error) {
	if err := s.validateChatPrompt(model, prompt); err != nil {
		return nil, err
	}

	text, err := s.genPrompt(prompt)
	if err != nil {
		return nil, err
	}

	body, err := json.Marshal(map[string]interface{}{
		"model":                s.getModelName(model),
		"prompt":               text,
		"max_tokens_to_sample": 4096,
		"temperature":          conf.Temperature,
		"stream":               true,
	})
	if err != nil {
		return nil, err
	}

	resp, err := s.do(ctx, "POST", "/v1/complete", body)
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

			var result struct {
				Completion string `json:"completion"`
				StopReason string `json:"stop_reason"`
				Model      string `json:"model"`
			}
			if err := json.NewDecoder(bytes.NewReader(ev.Data)).Decode(&result); err != nil {
				logrus.Errorf("failed to decode event: %v", err)
				return
			}

			if result.StopReason == "max_tokens" {
				logrus.Warn("max tokens reached")
				return
			}

			if result.StopReason == "stop_sequence" {
				return
			}

			ret <- &client.OpenAIMessage{
				Role:    "assistant",
				Content: result.Completion,
			}
		}
	}()

	return ret, nil
}

func (s *AnthropicStub) genPrompt(prompt []*client.OpenAIMessage) (string, error) {
	var texts []string
	for _, p := range prompt {
		switch p.GetRole() {
		case "system":
			texts = append(texts, fmt.Sprintf("Human: %s. Are you clear?", p.GetContent()))
			texts = append(texts, "Assistant: yes")
		case "assistant":
			texts = append(texts, fmt.Sprintf("Assistant: %s", p.GetContent()))
		case "user":
			texts = append(texts, fmt.Sprintf("Human: %s", p.GetContent()))
		default:
			return "", errors.Errorf("invalid role: %s", p.GetRole())
		}
	}

	texts = append(texts, "Assistant: ")
	texts = lo.Map(texts, func(item string, _ int) string {
		return "\n\n" + item
	})
	return strings.Join(texts, ""), nil
}

func (s *AnthropicStub) validateChatPrompt(model client.ModelType, prompt []*client.OpenAIMessage) error {
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
	text, err := s.genPrompt(prompt)
	if err != nil {
		return err
	}
	if s.MaxTokens(model) <= s.calcNumToken(text) {
		return errors.New("prompt too long")
	}

	return nil
}

func (s *AnthropicStub) Embeddings(ctx context.Context, model client.ModelType, texts []string) ([][]float32, *client.Usage, error) {
	return nil, nil, errors.New("not implemented")
}

// calcNumToken 函数根据给定的分词器和内容计算令牌数量
func (s *AnthropicStub) calcNumToken(content string) int {
	s.lock.Lock()
	defer s.lock.Unlock()
	ret, err := s.tk.EncodeSingle(content)
	if err != nil {
		logrus.Warnf("EncodeSingle: %s err=%v", content, err)
		return 0
	}
	return len(ret.GetTokens()) + 4
}
