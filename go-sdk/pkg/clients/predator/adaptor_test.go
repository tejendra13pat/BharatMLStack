package predator

import (
	"testing"

	triton "github.com/Meesho/BharatMLStack/go-sdk/pkg/clients/predator/client/grpc"
	"github.com/stretchr/testify/assert"
)

func TestMapPredatorRequestToProto_ReturnsNilWhenInputCannotBeConverted(t *testing.T) {
	a := &Adapter{}
	req := &PredatorRequest{
		ModelName: "model",
		Inputs: []Input{
			{Name: "in0", DataType: "FP64", Dims: []int{1}, StringData: [][][]string{{{"1.5"}}}},
		},
		BatchSize: 1,
		Deadline:  1000,
	}

	var got []*triton.ModelInferRequest
	assert.NotPanics(t, func() {
		got = a.MapPredatorRequestToProto(req, PredatorDataTypeString, []BatchInfo{{StartIndex: 0, EndIndex: 1}})
	})

	assert.Nil(t, got)
}
