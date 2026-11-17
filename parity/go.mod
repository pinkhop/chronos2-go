module github.com/pinkhop/chronos2-go/parity

go 1.27.4

require (
	github.com/pinkhop/chronos2-go v1.0.0
	github.com/pinkhop/chronos2-go-base v1.0.1
	github.com/pinkhop/chronos2-go-small v1.0.0
	golang.org/x/mod v0.43.0
)

require (
	github.com/pinkhop/chronos2-go-base-part1 v1.0.0 // indirect
	github.com/pinkhop/chronos2-go-base-part2 v1.0.0 // indirect
	github.com/pinkhop/chronos2-go-base-part3 v1.0.0 // indirect
	github.com/pinkhop/chronos2-go-base-part4 v1.0.0 // indirect
)

// parity is never published, so it may point at the root module's working
// tree directly. The model modules live in their own repositories and resolve
// from the module cache, or from a workspace that uses their checkouts.
replace github.com/pinkhop/chronos2-go => ../
