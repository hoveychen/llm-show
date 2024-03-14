package client

import (
	"strings"

	"github.com/pkg/errors"
)

type ModelType string

const (
	// Chat models
	ModelGpt3           ModelType = "gpt-3.5"
	ModelGpt4           ModelType = "gpt-4"
	ModelGpt4Vision     ModelType = "gpt-4v"
	ModelClaude2        ModelType = "claude-2"
	ModelClaudeInstant1 ModelType = "claude-instant-1"

	// Embedding models
	ModelTextEmbeddingAda ModelType = "text-embedding-ada"

	// Image models
	ModelDalle2   ModelType = "dalle-2"
	ModelDalle3   ModelType = "dalle-3"
	ModelDalle3HD ModelType = "dalle-3-hd"

	// Speech models
	ModelTTS1   ModelType = "tts-1"
	ModelTTS1HD ModelType = "tts-1-hd"

	// Deprecated models
	ModelGpt3_16k            ModelType = "gpt-3.5-16k"           // Deprecated.
	ModelGpt4_32k            ModelType = "gpt-4-32k"             // Deprecated.
	ModelTextDavinci         ModelType = "text-davinci"          // Deprecated.
	ModelClaude2_100k        ModelType = "claude-2-100k"         // Deprecated.
	ModelClaudeInstant1_100k ModelType = "claude-instant-1-100k" // Deprecated.
)

func ListChatModels() []ModelType {
	return []ModelType{
		ModelGpt3,
		ModelGpt4,
		ModelGpt4Vision,
		ModelClaude2,
		ModelClaudeInstant1,
	}
}

func ListEmbeddingModels() []ModelType {
	return []ModelType{
		ModelTextEmbeddingAda,
	}
}

func ListImageModels() []ModelType {
	return []ModelType{
		ModelDalle2,
		ModelDalle3,
		ModelDalle3HD,
	}
}

func ListSpeechModels() []ModelType {
	return []ModelType{
		ModelTTS1,
		ModelTTS1HD,
	}
}

func ParseModel(name string) (ModelType, error) {
	switch strings.ToLower(name) {
	case "gpt3", "gpt-3", "gpt-3.5", "gpt-3.5-turbo", "gpt-3.5-turbo-0301", "gpt-3.5-turbo-0613", "gpt-3.5-turbo-1106":
		return ModelGpt3, nil
	case "gpt3-16k", "gpt-3-16k", "gpt-3.5-16k", "gpt-3.5-turbo-16k", "gpt-3.5-turbo-16k-0613":
		return ModelGpt3, nil
	case "gpt4", "gpt-4", "gpt-4-0314", "gpt-4-0613", "gpt-4-1106-preview":
		return ModelGpt4, nil
	case "gpt4-32k", "gpt-4-32k", "gpt-4-32k-0314", "gpt-4-32k-0613":
		return ModelGpt4, nil
	case "gpt-4v", "gpt-4-vision-preview":
		return ModelGpt4Vision, nil
	case "claude1", "claude-1", "claude2", "claude-2":
		return ModelClaude2, nil
	case "claude1-100k", "claude-1-100k", "claude2-100k", "claude-2-100k":
		return ModelClaude2, nil
	case "claude-instant-1":
		return ModelClaudeInstant1, nil
	case "claude-instant-1-100k":
		return ModelClaudeInstant1, nil
	case "text-embedding-ada", "text-embedding-ada-002":
		return ModelTextEmbeddingAda, nil
	case "dalle-2", "dalle2":
		return ModelDalle2, nil
	case "dalle-3", "dalle3":
		return ModelDalle3, nil
	case "dalle-3-hd", "dalle3hd":
		return ModelDalle3HD, nil
	case "tts-1", "tts1":
		return ModelTTS1, nil
	case "tts-1-hd", "tts1hd":
		return ModelTTS1HD, nil
	default:
		return "", errors.Errorf("unknown model %q", name)
	}
}
