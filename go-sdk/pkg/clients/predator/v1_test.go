package predator

import (
	"context"
	"encoding/binary"
	"math"
	"strconv"
	"sync"
	"sync/atomic"
	"testing"

	triton "github.com/Meesho/BharatMLStack/go-sdk/pkg/clients/predator/client/grpc"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"
)

// fakeTriton is a minimal in-process Triton client. Only ModelInfer is implemented; any other
// method would panic through the embedded nil interface, which these tests never call.
type fakeTriton struct {
	triton.GRPCInferenceServiceClient
	infer func(req *triton.ModelInferRequest) (*triton.ModelInferResponse, error)
}

func (f *fakeTriton) ModelInfer(_ context.Context, in *triton.ModelInferRequest, _ ...grpc.CallOption) (*triton.ModelInferResponse, error) {
	return f.infer(in)
}

func newTestClient(infer func(req *triton.ModelInferRequest) (*triton.ModelInferResponse, error)) *ClientV1 {
	return &ClientV1{
		adapter:     Adapter{},
		callerId:    "caller",
		callerToken: "token",
		grpcClient:  &fakeTriton{infer: infer},
	}
}

func f32(v float32) []byte {
	b := make([]byte, 4)
	binary.LittleEndian.PutUint32(b, math.Float32bits(v))
	return b
}

func f64(v float64) []byte {
	b := make([]byte, 8)
	binary.LittleEndian.PutUint64(b, math.Float64bits(v))
	return b
}

// echoFirstInput answers every batch with one FP32 score per row, equal to the first float of that
// row in the first input, so tests can check that rows come back in request order.
func echoFirstInput(req *triton.ModelInferRequest) (*triton.ModelInferResponse, error) {
	rows := int(req.Inputs[0].Shape[0])
	raw := make([]byte, 0, rows*4)
	rowBytes := len(req.RawInputContents[0]) / rows
	for r := 0; r < rows; r++ {
		raw = append(raw, req.RawInputContents[0][r*rowBytes:r*rowBytes+4]...)
	}
	return &triton.ModelInferResponse{
		ModelName: req.ModelName,
		Outputs: []*triton.ModelInferResponse_InferOutputTensor{
			{Name: "out", Datatype: "FP32", Shape: []int64{int64(rows), 1}},
		},
		RawOutputContents: [][]byte{raw},
	}, nil
}

func rawFP32Rows(values ...float32) [][][]byte {
	rows := make([][][]byte, len(values))
	for i, v := range values {
		rows[i] = [][]byte{f32(v)}
	}
	return rows
}

func stringFP32Rows(values ...float32) [][][]string {
	rows := make([][][]string, len(values))
	for i, v := range values {
		rows[i] = [][]string{{strconv.FormatFloat(float64(v), 'f', -1, 32)}}
	}
	return rows
}

func baseRequest() *PredatorRequest {
	return &PredatorRequest{
		ModelName: "model",
		Inputs: []Input{
			{Name: "in0", DataType: "FP32", Dims: []int{1}, Data: rawFP32Rows(0, 1, 2, 3, 4)},
		},
		Outputs:   []Output{{Name: "out", ModelScores: []string{"score"}, Dims: [][]int{{1}}}},
		BatchSize: 2,
		Deadline:  1000,
	}
}

func fp32At(t *testing.T, b []byte) float32 {
	t.Helper()
	require.Len(t, b, 4)
	return math.Float32frombits(binary.LittleEndian.Uint32(b))
}

func TestGetInferenceScore_RawInputsAreBatchedAndMergedInOrder(t *testing.T) {
	var (
		mu        sync.Mutex
		batchRows []int64
	)
	client := newTestClient(func(req *triton.ModelInferRequest) (*triton.ModelInferResponse, error) {
		mu.Lock()
		batchRows = append(batchRows, req.Inputs[0].Shape[0])
		mu.Unlock()
		return echoFirstInput(req)
	})

	resp, err := client.GetInferenceScore(baseRequest())

	require.NoError(t, err)
	require.Len(t, resp.Outputs, 1)
	out := resp.Outputs[0]
	assert.Equal(t, "score", out.Name)
	assert.Equal(t, []int64{5, 1}, out.Shape)
	require.Len(t, out.Data, 5)
	for i := range out.Data {
		assert.Equal(t, float32(i), fp32At(t, out.Data[i]), "row %d", i)
	}
	assert.ElementsMatch(t, []int64{2, 2, 1}, batchRows)
}

func TestGetInferenceScore_StringInputs(t *testing.T) {
	client := newTestClient(echoFirstInput)
	req := baseRequest()
	req.Inputs[0].Data = nil
	req.Inputs[0].StringData = stringFP32Rows(0, 1, 2, 3, 4)

	resp, err := client.GetInferenceScore(req)

	require.NoError(t, err)
	require.Len(t, resp.Outputs, 1)
	assert.Equal(t, [][]string{{"0"}, {"1"}, {"2"}, {"3"}, {"4"}}, resp.Outputs[0].StringData)
}

func TestGetInferenceScoreV2_MergesBatchesInOrder(t *testing.T) {
	client := newTestClient(echoFirstInput)

	resp, err := client.GetInferenceScoreV2(baseRequest())

	require.NoError(t, err)
	require.Len(t, resp.RawOutputContents, 1)
	require.Len(t, resp.RawOutputContents[0], 5*4)
	assert.Equal(t, []int64{5, 1}, resp.Outputs[0].Shape)
	for i := 0; i < 5; i++ {
		assert.Equal(t, float32(i), fp32At(t, resp.RawOutputContents[0][i*4:(i+1)*4]), "row %d", i)
	}
}

// Malformed requests must come back as errors; they used to panic inside the SDK on the caller's
// goroutine (index out of range, nil pointer dereference).
func TestGetInferenceScore_InvalidRequestsReturnErrors(t *testing.T) {
	tests := []struct {
		name    string
		mutate  func(r *PredatorRequest)
		wantErr string
	}{
		{
			name: "second input has more rows than the first",
			mutate: func(r *PredatorRequest) {
				r.Inputs = append(r.Inputs, Input{Name: "in1", DataType: "FP32", Dims: []int{1}, Data: rawFP32Rows(0, 1, 2, 3, 4, 5, 6)})
			},
			wantErr: "in1",
		},
		{
			name: "second input has fewer rows than the first",
			mutate: func(r *PredatorRequest) {
				r.Inputs = append(r.Inputs, Input{Name: "in1", DataType: "FP32", Dims: []int{1}, Data: rawFP32Rows(0, 1)})
			},
			wantErr: "in1",
		},
		{
			name: "second input has no data while the first has",
			mutate: func(r *PredatorRequest) {
				r.Inputs = append(r.Inputs, Input{Name: "in1", DataType: "FP32", Dims: []int{1}})
			},
			wantErr: "in1",
		},
		{
			name: "inputs mix raw and string data",
			mutate: func(r *PredatorRequest) {
				r.Inputs = append(r.Inputs, Input{Name: "in1", DataType: "FP32", Dims: []int{1}, StringData: stringFP32Rows(0, 1, 2, 3, 4)})
			},
			wantErr: "in1",
		},
		{
			name: "row has fewer features than the first row",
			mutate: func(r *PredatorRequest) {
				r.Inputs[0].Data = [][][]byte{{f32(0), f32(0)}, {f32(1)}, {f32(2), f32(2)}}
			},
			wantErr: "in0",
		},
		{
			name: "first row has no features",
			mutate: func(r *PredatorRequest) {
				r.Inputs[0].Data = [][][]byte{{}, {}}
			},
			wantErr: "in0",
		},
		{
			name: "string input with a datatype that cannot be converted",
			mutate: func(r *PredatorRequest) {
				r.Inputs[0].Data = nil
				r.Inputs[0].DataType = "FP64"
				r.Inputs[0].StringData = [][][]string{{{"1.5"}}, {{"2.5"}}}
			},
			wantErr: "FP64",
		},
		{
			name: "string BYTES input with an empty feature",
			mutate: func(r *PredatorRequest) {
				r.Inputs[0].Data = nil
				r.Inputs[0].DataType = "BYTES"
				r.Inputs[0].StringData = [][][]string{{{}}, {{"a"}}}
			},
			wantErr: "in0",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var called atomic.Bool
			client := newTestClient(func(req *triton.ModelInferRequest) (*triton.ModelInferResponse, error) {
				called.Store(true)
				return echoFirstInput(req)
			})
			req := baseRequest()
			tt.mutate(req)

			var (
				resp *PredatorResponse
				err  error
			)
			require.NotPanics(t, func() { resp, err = client.GetInferenceScore(req) })
			require.NotPanics(t, func() { _, _ = client.GetInferenceScoreV2(req) })

			assert.Nil(t, resp)
			require.Error(t, err)
			assert.Contains(t, err.Error(), tt.wantErr)
			assert.False(t, called.Load(), "an invalid request must not reach the server")
		})
	}
}

func TestGetInferenceScore_OutputDatatypes(t *testing.T) {
	tests := []struct {
		datatype string
		raw      func(row int) []byte
		size     int
	}{
		{"FP64", func(r int) []byte { return f64(float64(r) + 0.5) }, 8},
		{"UINT8", func(r int) []byte { return []byte{byte(r)} }, 1},
		{"UINT16", func(r int) []byte { b := make([]byte, 2); binary.LittleEndian.PutUint16(b, uint16(r)); return b }, 2},
		{"UINT32", func(r int) []byte { b := make([]byte, 4); binary.LittleEndian.PutUint32(b, uint32(r)); return b }, 4},
		{"UINT64", func(r int) []byte { b := make([]byte, 8); binary.LittleEndian.PutUint64(b, uint64(r)); return b }, 8},
	}

	for _, tt := range tests {
		t.Run(tt.datatype, func(t *testing.T) {
			client := newTestClient(func(req *triton.ModelInferRequest) (*triton.ModelInferResponse, error) {
				rows := int(req.Inputs[0].Shape[0])
				var raw []byte
				for r := 0; r < rows; r++ {
					raw = append(raw, tt.raw(r)...)
				}
				return &triton.ModelInferResponse{
					Outputs: []*triton.ModelInferResponse_InferOutputTensor{
						{Name: "out", Datatype: tt.datatype, Shape: []int64{int64(rows), 1}},
					},
					RawOutputContents: [][]byte{raw},
				}, nil
			})
			req := baseRequest()
			req.BatchSize = 10

			var (
				resp *PredatorResponse
				err  error
			)
			require.NotPanics(t, func() { resp, err = client.GetInferenceScore(req) })

			require.NoError(t, err)
			require.Len(t, resp.Outputs, 1)
			require.Len(t, resp.Outputs[0].Data, 5)
			for r, b := range resp.Outputs[0].Data {
				assert.Equal(t, tt.raw(r), b, "row %d", r)
			}
			assert.Equal(t, tt.datatype, resp.Outputs[0].DataType)
		})
	}
}

func TestGetInferenceScore_UnknownOutputDatatypeReturnsError(t *testing.T) {
	client := newTestClient(func(req *triton.ModelInferRequest) (*triton.ModelInferResponse, error) {
		return &triton.ModelInferResponse{
			Outputs: []*triton.ModelInferResponse_InferOutputTensor{
				{Name: "out", Datatype: "FP128", Shape: []int64{1, 1}},
			},
			RawOutputContents: [][]byte{make([]byte, 16)},
		}, nil
	})
	req := baseRequest()
	req.BatchSize = 10

	var err error
	require.NotPanics(t, func() { _, err = client.GetInferenceScore(req) })

	assert.Error(t, err)
}

func TestGetInferenceScore_NoRequestedScoresDoesNotPanic(t *testing.T) {
	client := newTestClient(echoFirstInput)
	req := baseRequest()
	req.Outputs = []Output{{Name: "out"}}

	require.NotPanics(t, func() {
		resp, err := client.GetInferenceScore(req)
		assert.NoError(t, err)
		if assert.NotNil(t, resp) {
			assert.Empty(t, resp.Outputs)
		}
	})
}

func TestGetInferenceScore_InputWithoutDimsDoesNotPanic(t *testing.T) {
	client := newTestClient(echoFirstInput)
	req := baseRequest()
	req.Inputs[0].Dims = nil

	var (
		resp *PredatorResponse
		err  error
	)
	require.NotPanics(t, func() { resp, err = client.GetInferenceScore(req) })

	require.NoError(t, err)
	assert.Equal(t, []int64{5, 1}, resp.Outputs[0].Shape)
}
