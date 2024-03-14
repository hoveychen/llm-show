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
	"time"

	"github.com/hoveychen/mimic/pkg/client"
	"github.com/hoveychen/mimic/pkg/utils"
	"github.com/pkg/errors"
	"github.com/sirupsen/logrus"
	"github.com/sugarme/tokenizer"
	"github.com/sugarme/tokenizer/pretrained"
)

func init() {
	client.RegisterProvider("openai", NewOpenAIStub)
}

var _ client.Stub = (*OpenAIStub)(nil)
var _ client.StubWithFunctionCall = (*OpenAIStub)(nil)
var _ client.StubWithImageGeneration = (*OpenAIStub)(nil)
var _ client.StubWithSpeechGeneration = (*OpenAIStub)(nil)

type OpenAIStub struct {
	tk         *tokenizer.Tokenizer
	lock       sync.Mutex
	conf       *client.SubscriptionConfig
	concurrent chan struct{}
}

func NewOpenAIStub(conf *client.SubscriptionConfig) client.Stub {
	configFile, err := tokenizer.CachedPath("gpt2", "tokenizer.json")
	utils.PanicError(err)

	tk, err := pretrained.FromFile(configFile)
	utils.PanicError(err)

	return &OpenAIStub{
		conf:       conf,
		tk:         tk,
		concurrent: make(chan struct{}, conf.Concurrent),
	}
}

func (s *OpenAIStub) MaxTokens(model client.ModelType) int {
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

func (s *OpenAIStub) SupportFunctionCall() {}

func (s *OpenAIStub) do(ctx context.Context, method, path string, data []byte) (*http.Response, error) {
	s.concurrent <- struct{}{}
	defer func() {
		<-s.concurrent
	}()
	req, err := http.NewRequestWithContext(ctx, method, s.conf.Endpoint+path, bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	req.Header.Add("Content-Type", "application/json")
	req.Header.Add("Authorization", fmt.Sprintf("Bearer %s", s.conf.ApiSecret))

	return http.DefaultClient.Do(req)
}

func (s *OpenAIStub) getModelName(model client.ModelType) string {
	switch model {
	case client.ModelGpt3:
		return "gpt-3.5-turbo-1106"
	case client.ModelGpt4:
		return "gpt-4-1106-preview"
	case client.ModelGpt4Vision:
		return "gpt-4-vision-preview"
	case client.ModelTextDavinci:
		return "text-davinci-003"
	case client.ModelTextEmbeddingAda:
		return "text-embedding-ada-002"
	case client.ModelDalle2:
		return "dall-e-2"
	case client.ModelDalle3, client.ModelDalle3HD:
		return "dall-e-3"
	case client.ModelTTS1:
		return "tts-1"
	case client.ModelTTS1HD:
		return "tts-1-hd"
	default:
		return ""
	}
}

func (s *OpenAIStub) supportVision(model client.ModelType) bool {
	switch model {
	case client.ModelGpt4Vision:
		return true
	default:
		return false
	}
}

func (s *OpenAIStub) supportTools(model client.ModelType) bool {
	switch model {
	case client.ModelGpt3, client.ModelGpt4:
		return true
	default:
		return false
	}
}

func (s *OpenAIStub) newChatRequest(model client.ModelType, prompt []*client.OpenAIMessage, conf *client.CallConfig) map[string]interface{} {
	reqData := map[string]interface{}{
		"model":             s.getModelName(model),
		"messages":          s.newRequestMessages(model, prompt, conf.ExtraAttention),
		"temperature":       conf.Temperature,
		"max_tokens":        2048,
		"frequency_penalty": 0,
		"presence_penalty":  0,
		"top_p":             1,
	}
	if conf.RespFormat == "json_object" {
		reqData["response_format"] = map[string]string{"type": "json_object"}
	}

	if s.supportTools(model) {
		if len(conf.Functions) > 0 {
			if s.conf.ParallelFunctionCall {
				reqData["tools"] = s.genTools(conf.Functions)
			} else {
				reqData["functions"] = s.genFunctions(conf.Functions)
			}
		}
		if conf.FunctionCall != "" {
			if s.conf.ParallelFunctionCall {
				reqData["tool_choice"] = map[string]interface{}{
					"type": "function",
					"function": map[string]string{
						"name": conf.FunctionCall,
					},
				}
			} else {
				reqData["function_call"] = map[string]string{
					"name": conf.FunctionCall,
				}
			}
		}
	}
	return reqData
}

func (s *OpenAIStub) newRequestMessages(model client.ModelType, prompt []*client.OpenAIMessage, extraAttention bool) []map[string]interface{} {
	var ret []map[string]interface{}
	var sysRet []map[string]interface{}

	for _, p := range prompt {
		msg := map[string]interface{}{
			"role": p.GetRole(),
		}
		// TODO(yuheng): Seems to be a bug of openai:
		// https://community.openai.com/t/example-code-in-function-calling-is-erroneous-in-step-4-complaining-content-is-a-required-property-messages-1/265061/4
		content := p.GetContent()
		if content == "" {
			content = "null"
		}

		switch p.GetRole() {
		case "user":
			if s.supportVision(model) {
				type imageData struct {
					URL    string `json:"url"`
					Detail string `json:"detail"`
				}
				type contentData struct {
					Type     string    `json:"type"`
					Text     string    `json:"text,omitempty"`
					ImageURL imageData `json:"image_url,omitempty"`
				}
				var contents []*contentData
				contents = append(contents, &contentData{
					Type: "text",
					Text: content,
				})
				for _, url := range p.GetImageURLs() {
					contents = append(contents, &contentData{
						Type: "image_url",
						ImageURL: imageData{
							URL:    url,
							Detail: "low",
						},
					})
				}
				msg["content"] = contents
			} else {
				msg["content"] = content
			}
		case "assistant":
			if !s.supportTools(model) {
				if p.GetFunctionCall().GetName() != "" {
					content += "Requesting function call: " + p.GetFunctionCall().GetName()
				}
				for _, tool := range p.GetToolCalls() {
					content += "Requesting tool call: " + tool.GetFunction().GetName()
				}
				msg["content"] = content
				break
			}
			msg["content"] = content
			if p.GetFunctionCall().GetName() != "" {
				msg["function_call"] = p.GetFunctionCall()
			}
			if len(p.GetToolCalls()) > 0 {
				msg["tool_calls"] = p.GetToolCalls()
			}
		case "system":
			msg["content"] = content
		case "tool":
			if !s.supportTools(model) {
				msg["role"] = "assistant"
				msg["content"] = p.GetContent()
				break
			}
			msg["content"] = content
			msg["tool_call_id"] = p.GetToolCallID()
		case "function":
			if !s.supportTools(model) {
				msg["role"] = "assistant"
				msg["content"] = content
				break
			}
			msg["content"] = content
			if p.GetName() != "" {
				msg["name"] = p.GetName()
			}
		}
		if p.GetRole() == "system" {
			sysRet = append(sysRet, msg)
		} else {
			ret = append(ret, msg)
		}
	}
	// Reorder the system messages to the end.
	// Todo(yuheng): It seems the order of system messages will affect the modernaity of the response.
	if extraAttention {
		ret = append(ret, sysRet...)
		return ret
	}
	sysRet = append(sysRet, ret...)
	return sysRet
}

func (s *OpenAIStub) Chat(ctx context.Context, model client.ModelType, conf *client.CallConfig, prompt []*client.OpenAIMessage) (*client.OpenAIMessage, *client.Usage, error) {
	if err := s.validateChatPrompt(model, prompt); err != nil {
		return nil, nil, err
	}

	reqData := s.newChatRequest(model, prompt, conf)

	body, err := json.Marshal(reqData)
	if err != nil {
		return nil, nil, err
	}

	resp, err := s.do(ctx, "POST", "/v1/chat/completions", body)
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
			Message      *client.OpenAIMessage `json:"message"` // Always role=assistant
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

func (s *OpenAIStub) ChatStream(ctx context.Context, model client.ModelType, conf *client.CallConfig, prompt []*client.OpenAIMessage) (chan *client.OpenAIMessage, error) {
	if err := s.validateChatPrompt(model, prompt); err != nil {
		return nil, err
	}

	reqData := s.newChatRequest(model, prompt, conf)
	reqData["stream"] = true

	body, err := json.Marshal(reqData)
	if err != nil {
		return nil, err
	}

	resp, err := s.do(ctx, "POST", "/v1/chat/completions", body)
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

			if result.Choices[0].FinishReason == "stop" || result.Choices[0].FinishReason == "function_call" {
				return
			}

			ret <- result.Choices[0].Delta
		}
	}()

	return ret, nil
}

type openAITool struct {
	Type     string          `json:"type"` // Only "function" is supported.
	Function *openAIFunction `json:"function"`
}

type openAIFunction struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  openAIParamter `json:"parameters,omitempty"`
}

type openAIParamter struct {
	Description string                     `json:"description,omitempty"`
	Type        string                     `json:"type"`
	Properties  map[string]*openAIParamter `json:"properties,omitempty"`
	Required    []string                   `json:"required,omitempty"`
}

func (s *OpenAIStub) genFunctions(fn []*client.Function) []*openAIFunction {
	var ret []*openAIFunction
	for _, f := range fn {
		of := &openAIFunction{
			Name:        f.GetName(),
			Description: f.GetDescription(),
		}
		ret = append(ret, of)

		if len(f.GetFields()) == 0 {
			continue
		}
		of.Parameters.Type = "object"
		of.Parameters.Properties = make(map[string]*openAIParamter)
		for _, p := range f.GetFields() {
			of.Parameters.Properties[p.GetName()] = &openAIParamter{
				Description: p.GetDescription(),
				Type:        "string",
			}
			if p.GetRequired() {
				of.Parameters.Required = append(of.Parameters.Required, p.GetName())
			}
		}
	}
	return ret
}

func (s *OpenAIStub) genTools(fn []*client.Function) []*openAITool {
	var ret []*openAITool
	for _, f := range fn {
		of := &openAIFunction{
			Name:        f.GetName(),
			Description: f.GetDescription(),
		}
		tool := &openAITool{
			Type:     "function",
			Function: of,
		}
		ret = append(ret, tool)

		if len(f.GetFields()) == 0 {
			continue
		}
		of.Parameters.Type = "object"
		of.Parameters.Properties = make(map[string]*openAIParamter)
		for _, p := range f.GetFields() {
			of.Parameters.Properties[p.GetName()] = &openAIParamter{
				Description: p.GetDescription(),
				Type:        "string",
			}
			if p.GetRequired() {
				of.Parameters.Required = append(of.Parameters.Required, p.GetName())
			}
		}
	}
	return ret
}

func (s *OpenAIStub) validateChatPrompt(model client.ModelType, prompt []*client.OpenAIMessage) error {
	var textSlices []string
	for i, p := range prompt {
		switch p.GetRole() {
		case "system", "assistant", "user", "function", "tool":
		default:
			return errors.Errorf("invalid role: %s", p.GetRole())
		}
		if (p.GetContent() == "" && len(p.GetImageURLs()) == 0) && p.GetRole() != "assistant" {
			return errors.Errorf("empty prompt, idx: %d", i)
		}
		textSlices = append(textSlices, p.GetContent())
	}

	if s.MaxTokens(model) <= s.calcNumToken(strings.Join(textSlices, ";")) {
		return errors.Errorf("text too long.")
	}

	return nil
}

func (s *OpenAIStub) Embeddings(ctx context.Context, model client.ModelType, texts []string) ([][]float32, *client.Usage, error) {
	if err := s.validateEmbeddings(model, texts); err != nil {
		return nil, nil, err
	}

	body, err := json.Marshal(map[string]interface{}{
		"input": texts,
		"model": s.getModelName(model),
	})
	if err != nil {
		return nil, nil, err
	}

	resp, err := s.do(ctx, "POST", "/v1/embeddings", body)
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

	var ret [][]float32
	for _, data := range result.Data {
		ret = append(ret, data.Embedding)
	}

	usage := &client.Usage{
		Subscription: s.conf.Name,
		Model:        result.Model,
		PromptTokens: result.Usage.PromptTokens,
	}

	return ret, usage, nil
}

func (s *OpenAIStub) validateEmbeddings(model client.ModelType, texts []string) error {
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
func (s *OpenAIStub) calcNumToken(content string) int {
	s.lock.Lock()
	defer s.lock.Unlock()
	ret, err := s.tk.EncodeSingle(content)
	if err != nil {
		logrus.Warnf("EncodeSingle: %s err=%v", content, err)
		return 0
	}
	return len(ret.GetTokens()) + 4
}

func (s *OpenAIStub) GenerateImage(ctx context.Context, modelName client.ModelType, spec *client.ImageSpec) (*client.Image, error) {
	if err := s.validateImageSpec(modelName, spec); err != nil {
		return nil, err
	}

	req := map[string]interface{}{
		"prompt": spec.GetPrompt(),
		"model":  s.getModelName(modelName),
	}

	if modelName == client.ModelDalle3HD {
		req["quality"] = "hd"
	}
	if modelName == client.ModelDalle3 || modelName == client.ModelDalle3HD {
		style := spec.GetStyle()
		if style != "vivid" && style != "natural" {
			style = "vivid"
		}
		req["style"] = style
	}

	body, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}

	since := time.Now()

	resp, err := s.do(ctx, "POST", "/v1/images/generations", body)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		if strings.Contains(string(body), "429") {
			return nil, client.ErrTooManyRequests
		}
		return nil, fmt.Errorf("unexpected status : %s %s", resp.Status, string(body))
	}

	var result struct {
		Created int64 `json:"created"`
		Data    []struct {
			B64JSON       string `json:"b64_json"`
			URL           string `json:"url"`
			RevisedPrompt string `json:"revised_prompt"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}
	if len(result.Data) == 0 {
		return nil, fmt.Errorf("no image")
	}

	logrus.Infof("generated image: %s elapse=%s", utils.JsonifyNested(req), time.Since(since))

	image := result.Data[0]
	return &client.Image{
		URL:           image.URL,
		RevisedPrompt: image.RevisedPrompt,
	}, nil
}

func (s *OpenAIStub) validateImageSpec(model client.ModelType, spec *client.ImageSpec) error {
	if strings.TrimSpace(spec.GetPrompt()) == "" {
		return errors.New("empty text")
	}

	var textLimit int
	switch model {
	case client.ModelDalle2:
		textLimit = 1000
	case client.ModelDalle3, client.ModelDalle3HD:
		textLimit = 4000
	default:
		return errors.Errorf("unsupported model: %s", model)
	}

	if len(spec.GetPrompt()) > textLimit {
		return errors.Errorf("text too long. limit=%d", textLimit)
	}

	return nil
}

func (s *OpenAIStub) GenerateSpeech(ctx context.Context, modelName client.ModelType, spec *client.SpeechSpec) (*client.Speech, error) {
	if err := s.validateSpeechSpec(modelName, spec); err != nil {
		return nil, err
	}

	voice := spec.GetVoice()
	if voice == "" {
		voice = "fable"
	}

	speed := spec.GetSpeed()
	if speed == 0 {
		speed = 1.0
	}

	req := map[string]interface{}{
		"input":           spec.GetText(),
		"model":           s.getModelName(modelName),
		"voice":           voice,
		"speed":           speed,
		"response_format": "mp3",
	}

	body, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}

	since := time.Now()
	resp, err := s.do(ctx, "POST", "/v1/audio/speech", body)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	respBody, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		if strings.Contains(string(body), "429") {
			return nil, client.ErrTooManyRequests
		}
		return nil, fmt.Errorf("unexpected status : %s %s", resp.Status, string(body))
	}

	if len(respBody) == 0 {
		return nil, fmt.Errorf("no speech")
	}

	logrus.Infof("generated speech: %s elapse=%s", utils.JsonifyNested(req), time.Since(since))

	return &client.Speech{
		Audio: respBody,
	}, nil
}

func (s *OpenAIStub) validateSpeechSpec(model client.ModelType, spec *client.SpeechSpec) error {
	if strings.TrimSpace(spec.GetText()) == "" {
		return errors.New("empty text")
	}

	var textLimit int
	var availableVoices []string
	switch model {
	case client.ModelTTS1, client.ModelTTS1HD:
		textLimit = 4096
		availableVoices = []string{"alloy", "echo", "fable", "onyx", "nova", "shimmer"}
	default:
		return errors.Errorf("unsupported model: %s", model)
	}

	if len(spec.GetText()) > textLimit {
		return errors.Errorf("text too long. limit=%d", textLimit)
	}

	if spec.GetVoice() != "" && !utils.Contain(availableVoices, spec.GetVoice()) {
		return errors.Errorf("unsupported voice: %s", spec.GetVoice())
	}

	if spec.GetSpeed() > 0 && (spec.GetSpeed() < 0.25 || spec.GetSpeed() > 4.0) {
		return errors.Errorf("unsupported speed: %f", spec.GetSpeed())
	}

	return nil
}
