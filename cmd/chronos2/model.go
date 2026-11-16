package main

import (
	"context"
	"fmt"

	chronos2 "github.com/pinkhop/chronos2-go"
	base "github.com/pinkhop/chronos2-go-base"
	small "github.com/pinkhop/chronos2-go-small"
	"github.com/pinkhop/chronos2-go/cmd/chronos2/internal/command"
)

// forecastWithModel loads the requested model and forecasts the request's
// history. Only the requested model is decoded; the other stays embedded and
// untouched.
func forecastWithModel(ctx context.Context, request command.Request) (command.Forecast, error) {
	model, err := loadModel(request.Model)
	if err != nil {
		return nil, err
	}
	var options []chronos2.ForecastOption
	if request.ContextLength > 0 {
		options = append(options, chronos2.WithContextLength(request.ContextLength))
	}
	if request.LongHorizon {
		options = append(options, chronos2.WithLongHorizon())
	}
	forecast, err := model.Forecast(ctx, request.History, request.Horizon, options...)
	if err != nil {
		// Library errors already wrap a public sentinel with the offending
		// input named, so a further prefix would add only noise.
		return nil, withFlagHints(err)
	}
	return forecast, nil
}

// loadModel decodes the named embedded model; name is command.ModelSmall or
// command.ModelBase.
func loadModel(name string) (*chronos2.Model, error) {
	var model *chronos2.Model
	var err error
	switch name {
	case command.ModelSmall:
		model, err = small.New()
	case command.ModelBase:
		model, err = base.New()
	default:
		return nil, fmt.Errorf("%w: unknown model %q", command.ErrInvalidInput, name)
	}
	if err != nil {
		return nil, fmt.Errorf("load %s model: %w", name, err)
	}
	return model, nil
}
