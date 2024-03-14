package client

import (
	"context"

	"github.com/pkg/errors"
)

var ErrTooManyRequests = errors.New("too many requests")

// TODO(yuheng): Refactor the stub interface to decouple the functions. Not all stub provide chat function.
type Stub interface {
	Chat(context.Context, ModelType, *CallConfig, []*OpenAIMessage) (*OpenAIMessage, *Usage, error)
	ChatStream(context.Context, ModelType, *CallConfig, []*OpenAIMessage) (chan *OpenAIMessage, error)
	Embeddings(context.Context, ModelType, []string) ([][]float32, *Usage, error)
	MaxTokens(ModelType) int
}

type StubWithFunctionCall interface {
	Stub
	SupportFunctionCall()
}

type StubWithImageGeneration interface {
	Stub
	GenerateImage(context.Context, ModelType, *ImageSpec) (*Image, error)
}

type StubWithSpeechGeneration interface {
	Stub
	GenerateSpeech(context.Context, ModelType, *SpeechSpec) (*Speech, error)
}

func SupportFunctionCall(stub Stub) bool {
	_, ok := stub.(StubWithFunctionCall)
	return ok
}

func SupportImageGeneration(stub Stub) bool {
	_, ok := stub.(StubWithImageGeneration)
	return ok
}

func SupportSpeechGeneration(stub Stub) bool {
	_, ok := stub.(StubWithSpeechGeneration)
	return ok
}
