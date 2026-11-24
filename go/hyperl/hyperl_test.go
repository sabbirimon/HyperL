// SPDX-License-Identifier: Apache-2.0
package hyperl

import (
	"math"
	"testing"
)

func TestContextAndVersion(t *testing.T) {
	v := Version()
	if v == "" {
		t.Fatal("expected non-empty engine version")
	}
	ctx, err := NewContext(2)
	if err != nil {
		t.Fatalf("failed to create context: %v", err)
	}
	defer ctx.Close()
}

func TestRMSNorm(t *testing.T) {
	ctx, err := NewContext(1)
	if err != nil {
		t.Fatal(err)
	}
	defer ctx.Close()

	x := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	out, err := ctx.RMSNorm(x, nil, 1e-5)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 8 {
		t.Fatalf("expected 8 elements, got %d", len(out))
	}
	if math.Abs(float64(out[0]-0.1980295)) > 1e-4 {
		t.Fatalf("unexpected out[0]: %f", out[0])
	}
}

func TestGEMV(t *testing.T) {
	ctx, err := NewContext(2)
	if err != nil {
		t.Fatal(err)
	}
	defer ctx.Close()

	W := make([]float32, 4*8)
	for i := range W {
		W[i] = 1.0
	}
	x := []float32{1, 2, 3, 4, 5, 6, 7, 8} // sum = 36

	y, err := ctx.GEMV(W, x, 4, 8)
	if err != nil {
		t.Fatal(err)
	}
	if len(y) != 4 {
		t.Fatalf("expected 4 outputs, got %d", len(y))
	}
	for i, val := range y {
		if math.Abs(float64(val-36.0)) > 1e-4 {
			t.Fatalf("y[%d] expected 36, got %f", i, val)
		}
	}
}
