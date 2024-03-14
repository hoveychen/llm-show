package utils

// Contain return true if given string is in the string slice.
func Contain[T comparable](slice []T, s T) bool {
	for _, item := range slice {
		if item == s {
			return true
		}
	}
	return false
}
