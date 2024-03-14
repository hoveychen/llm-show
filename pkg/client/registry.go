package client

var (
	providerRegistry = map[string]StubConstructor{}
)

type StubConstructor func(*SubscriptionConfig) Stub

func RegisterProvider(provider string, constructor StubConstructor) {
	providerRegistry[provider] = constructor
}
