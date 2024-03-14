package client

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/hoveychen/mimic/pkg/utils"
	"github.com/pkg/errors"
	"github.com/sirupsen/logrus"
	"github.com/sugarme/tokenizer"
	"github.com/sugarme/tokenizer/pretrained"
)

type ClientConfig struct {
	MaxRetry       int
	ChatModel      string
	EmbeddingModel string
	Temperature    float32
	// How many tokens reserved for generating the result.
	MinOutputToken int
	Subscriptions  []*SubscriptionConfig
	// Default subscription for each model.
	// Key: model, Value: subscription name
	DefaultSubscription map[string]string
}

type SubscriptionConfig struct {
	Name                 string
	Provider             string // "azure", "openai", "anthropic"
	ApiSecret            string
	Endpoint             string
	Concurrent           int
	CensoredWords        []string // Azure only, case insenstive
	ParallelFunctionCall bool     // OpenAI only.

}

func (sc *SubscriptionConfig) SetDefault() {
	if sc.Concurrent == 0 {
		sc.Concurrent = 10
	}
}

func (cc *ClientConfig) SetDefault() {
	if cc.MaxRetry == 0 {
		cc.MaxRetry = 10
	}
	if cc.ChatModel == "" {
		cc.ChatModel = "gpt-3.5-turbo"
	}
	if cc.EmbeddingModel == "" {
		cc.EmbeddingModel = "text-embedding-ada-002"
	}
	if cc.DefaultSubscription == nil {
		cc.DefaultSubscription = make(map[string]string)
	}
	if cc.MinOutputToken == 0 {
		cc.MinOutputToken = 256
	}
	for _, sub := range cc.Subscriptions {
		sub.SetDefault()
	}
}

func (bc *ClientConfig) Validate() error {
	if len(bc.Subscriptions) == 0 {
		return errors.New("no subscription")
	}
	return nil
}

func (cc *ClientConfig) ChooseSubscription(model ModelType) (*SubscriptionConfig, error) {
	if cc.DefaultSubscription[string(model)] != "" {
		return cc.GetSubscription(cc.DefaultSubscription[string(model)])
	}
	// Return default first subscription.
	return cc.Subscriptions[0], nil
}

func (cc *ClientConfig) GetSubscription(name string) (*SubscriptionConfig, error) {
	for _, sub := range cc.Subscriptions {
		if sub.Name == name {
			return sub, nil
		}
	}
	return nil, fmt.Errorf("subscription %s not found", name)
}

type Client struct {
	conf  *ClientConfig
	stubs map[string]Stub
	tk    *tokenizer.Tokenizer
	lock  sync.Mutex
}

func NewClient(conf *ClientConfig) *Client {
	stubs := make(map[string]Stub)
	for _, sub := range conf.Subscriptions {
		constructor := providerRegistry[sub.Provider]
		if constructor == nil {
			utils.PanicError(fmt.Errorf("provider %s not found", sub.Provider))
		}
		stubs[sub.Name] = constructor(sub)
	}

	configFile, err := tokenizer.CachedPath("gpt2", "tokenizer.json")
	utils.PanicError(err)

	tk, err := pretrained.FromFile(configFile)
	utils.PanicError(err)

	return &Client{
		conf:  conf,
		stubs: stubs,
		tk:    tk,
	}
}

// TODO(yuheng): Refactor the struct to pass into the chat function.
type CallConfig struct {
	Model          string
	Temperature    float32
	ExtraAttention bool
	FunctionCall   string
	Functions      []*Function
	MaxRetry       int
	Subscription   *SubscriptionConfig
	Client         *ClientConfig
	RespFormat     string // OpenAI only, Must be one of text or json_object
}

type CallOption func(*CallConfig)

func WithModel(model string) CallOption {
	return func(conf *CallConfig) {
		conf.Model = model
	}
}

func WithTemperature(temperature float32) CallOption {
	return func(conf *CallConfig) {
		conf.Temperature = temperature
	}
}

func WithFunctionCall(functionCall string) CallOption {
	return func(conf *CallConfig) {
		conf.FunctionCall = functionCall
	}
}

func WithFunctions(functions []*Function) CallOption {
	return func(conf *CallConfig) {
		conf.Functions = functions
	}
}

func EnableJSONResp() CallOption {
	return func(conf *CallConfig) {
		conf.RespFormat = "json_object"
	}
}

func WithSubscription(subscription string) CallOption {
	return func(conf *CallConfig) {
		sub, err := conf.Client.GetSubscription(subscription)
		if err != nil {
			logrus.Warnf("subscription %s not found", subscription)
			return
		}
		conf.Subscription = sub
	}
}

func WithExtraAttention() CallOption {
	return func(conf *CallConfig) {
		conf.ExtraAttention = true
	}
}

func (b *Client) GetChatModel() string {
	return b.conf.ChatModel
}

func (b *Client) GetEmbeddingModel() string {
	return b.conf.EmbeddingModel
}

type Usage struct {
	Model            string `json:"model"`
	Subscription     string `json:"subscription"`
	PromptTokens     int    `json:"prompt_tokens"`
	CompletionTokens int    `json:"completion_tokens"`
}

func (c *Client) Chat(ctx context.Context, prompt []*OpenAIMessage, opts ...CallOption) (*OpenAIMessage, *Usage, error) {
	conf, stub, model, err := c.prepareChatRequest(ctx, opts...)
	if err != nil {
		return nil, nil, err
	}

	maxToken := stub.MaxTokens(model) - c.conf.MinOutputToken
	if SupportFunctionCall(stub) {
		maxToken -= c.estimateFunctionCallToken(conf.Functions)
	}

	prompt, err = c.filterConverstaion(maxToken, prompt)
	if err != nil {
		return nil, nil, err
	}

	var reply *OpenAIMessage
	var usage *Usage
	for attempt := 0; attempt < conf.MaxRetry; attempt++ {
		reply, usage, err = stub.Chat(ctx, model, conf, prompt)
		if err == nil {
			return reply, usage, nil
		}
		if err != ErrTooManyRequests {
			return nil, nil, err
		}
		logrus.Warnf("Chat failure error=%s model=%s subscription=%s attempt=%d", err, model, conf.Subscription.Name, attempt+1)
		time.Sleep(time.Second * time.Duration(attempt+1) * 2)
	}

	return nil, nil, errors.New("exceeded max retry")
}

func (c *Client) ChatStream(ctx context.Context, prompt []*OpenAIMessage, opts ...CallOption) (chan *OpenAIMessage, error) {
	conf, stub, model, err := c.prepareChatRequest(ctx, opts...)
	if err != nil {
		return nil, err
	}

	maxToken := stub.MaxTokens(model) - c.conf.MinOutputToken
	if SupportFunctionCall(stub) {
		maxToken -= c.estimateFunctionCallToken(conf.Functions)
	}

	prompt, err = c.filterConverstaion(maxToken, prompt)
	if err != nil {
		return nil, err
	}

	return stub.ChatStream(ctx, model, conf, prompt)
}

// filterConverstaion 方法根据最大会话令牌数过滤会话
func (c *Client) filterConverstaion(maxTokens int, messages []*OpenAIMessage) ([]*OpenAIMessage, error) {
	tokenRemain := maxTokens

	// always reserve system messages
	var sysMsgs []*OpenAIMessage
	for _, m := range messages {
		if m.GetRole() == "system" {
			sysMsgs = append(sysMsgs, m)
			tokenRemain -= c.estimateToken(m)
		}
	}
	if tokenRemain < 0 {
		return nil, errors.New("system prompt too long")
	}

	var conv []*OpenAIMessage
	for i := len(messages) - 1; i >= 0; i-- {
		m := messages[i]
		if m.GetRole() == "system" {
			continue
		}

		token := c.estimateToken(m)
		if tokenRemain-token < 0 {
			break
		}

		conv = append(conv, m)
	}
	// 反转顺序
	for i := len(conv)/2 - 1; i >= 0; i-- {
		opp := len(conv) - 1 - i
		conv[i], conv[opp] = conv[opp], conv[i]
	}

	var ret []*OpenAIMessage
	ret = append(ret, sysMsgs...)
	ret = append(ret, conv...)

	return ret, nil
}

func (c *Client) getTokens(data string) (*tokenizer.Encoding, error) {
	c.lock.Lock()
	defer c.lock.Unlock()
	return c.tk.EncodeSingle(data)
}

func (c *Client) estimateFunctionCallToken(functions []*Function) int {
	if len(functions) == 0 {
		return 0
	}
	var data []string
	for _, f := range functions {
		data = append(data, f.GetName(), f.GetDescription())
		for _, field := range f.GetFields() {
			data = append(data, field.GetName(), field.GetDescription())
		}
	}
	ret, err := c.getTokens(strings.Join(data, "  "))
	if err != nil {
		logrus.Warnf("EncodeSingle: %s err=%v", data, err)
		return 0
	}
	return len(ret.GetTokens()) + 4
}

// calcNumToken 函数根据给定的分词器和内容计算令牌数量
func (c *Client) estimateToken(msg *OpenAIMessage) int {
	var data string
	if msg.GetContent() != "" {
		data += msg.Content
	}

	data += strings.Join(msg.GetImageURLs(), " ")

	ret, err := c.getTokens(data)
	if err != nil {
		logrus.Warnf("EncodeSingle: %s err=%v", data, err)
		return 0
	}
	return len(ret.GetTokens()) + 4
}

func (c *Client) prepareChatRequest(ctx context.Context, opts ...CallOption) (*CallConfig, Stub, ModelType, error) {
	if ctx.Err() != nil {
		return nil, nil, "", ctx.Err()
	}

	conf := &CallConfig{
		Model:       c.conf.ChatModel,
		Temperature: c.conf.Temperature,
		MaxRetry:    c.conf.MaxRetry,
		Client:      c.conf,
	}
	for _, opt := range opts {
		opt(conf)
	}

	model, err := ParseModel(conf.Model)
	if err != nil {
		return nil, nil, "", err
	}
	if conf.Subscription == nil {
		sub, err := conf.Client.ChooseSubscription(model)
		if err != nil {
			return nil, nil, "", err
		}
		conf.Subscription = sub
	}

	stub := c.stubs[conf.Subscription.Name]
	if stub == nil {
		return nil, nil, "", fmt.Errorf("subscription %s not found", conf.Subscription.Name)
	}

	return conf, stub, model, nil
}

func (c *Client) Embeddings(ctx context.Context, texts []string, opts ...CallOption) ([][]float32, *Usage, error) {
	if ctx.Err() != nil {
		return nil, nil, ctx.Err()
	}

	conf := &CallConfig{
		Model:    c.conf.EmbeddingModel,
		MaxRetry: c.conf.MaxRetry,
		Client:   c.conf,
	}
	for _, opt := range opts {
		opt(conf)
	}
	model, err := ParseModel(conf.Model)
	if err != nil {
		return nil, nil, err
	}
	if conf.Subscription == nil {
		sub, err := conf.Client.ChooseSubscription(model)
		if err != nil {
			return nil, nil, err
		}
		conf.Subscription = sub
	}

	stub := c.stubs[conf.Subscription.Name]
	if stub == nil {
		return nil, nil, fmt.Errorf("subscription %s not found", conf.Subscription.Name)
	}

	for attempt := 0; attempt < conf.MaxRetry; attempt++ {
		reply, usage, err := stub.Embeddings(ctx, model, texts)
		if err == nil {
			return reply, usage, nil
		}
		if err != ErrTooManyRequests {
			return nil, nil, err
		}
		logrus.Warnf("Embedding failure error=%s model=%s subscription=%s attempt=%d", err, model, conf.Subscription.Name, attempt+1)
		time.Sleep(time.Second * time.Duration(attempt+1) * 2)
	}

	return nil, nil, errors.New("exceeded max retry")
}

func (c *Client) GenerateImage(ctx context.Context, modelName string, spec *ImageSpec) (*Image, error) {
	model, err := ParseModel(modelName)
	if err != nil {
		return nil, err
	}
	sub, err := c.conf.ChooseSubscription(model)
	if err != nil {
		return nil, err
	}

	stub := c.stubs[sub.Name]
	if !SupportImageGeneration(stub) {
		return nil, errors.New("image generation not supported")
	}

	generator := stub.(StubWithImageGeneration)
	return generator.GenerateImage(ctx, model, spec)
}

func (c *Client) GenerateSpeech(ctx context.Context, modelName string, spec *SpeechSpec) (*Speech, error) {
	model, err := ParseModel(modelName)
	if err != nil {
		return nil, err
	}
	sub, err := c.conf.ChooseSubscription(model)
	if err != nil {
		return nil, err
	}

	stub := c.stubs[sub.Name]
	if !SupportSpeechGeneration(stub) {
		return nil, errors.New("speech generation not supported")
	}

	generator := stub.(StubWithSpeechGeneration)
	return generator.GenerateSpeech(ctx, model, spec)
}
