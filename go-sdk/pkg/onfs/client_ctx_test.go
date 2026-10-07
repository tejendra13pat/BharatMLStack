package onfs

import (
	"context"
	"testing"
	"time"

	pb "github.com/Meesho/BharatMLStack/go-sdk/pkg/proto/onfs/retrieve"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"google.golang.org/grpc/metadata"
)

type ctxKey struct{}

func newCtxTestClient(m *mockRetrieveServiceClient, deadlineMs int64) *ClientV1 {
	return &ClientV1{
		v1Client: &clientConfig{
			client:   m,
			deadline: deadlineMs,
		},
		adapter:     Adapter{},
		batchSize:   10,
		callerId:    "test-caller",
		callerToken: "test-token",
	}
}

func ctxTestQuery() *Query {
	return &Query{
		EntityLabel: "test_entity",
		KeysSchema:  []string{"id"},
		Keys:        []Keys{{Cols: []string{"k1"}}},
		FeatureGroups: []FeatureGroup{
			{Label: "fg", FeatureLabels: []string{"f1"}},
		},
	}
}

func okRetrieveResult() *pb.Result {
	return &pb.Result{
		KeysSchema: []string{"id"},
		FeatureSchemas: []*pb.FeatureSchema{
			{FeatureGroupLabel: "fg", Features: []*pb.Feature{{Label: "f1", ColumnIdx: 0}}},
		},
		Rows: []*pb.Row{{Keys: []string{"k1"}, Columns: [][]byte{[]byte("v1")}}},
	}
}

func okDecodedResult() *pb.DecodedResult {
	return &pb.DecodedResult{
		KeysSchema: []string{"id"},
		FeatureSchemas: []*pb.FeatureSchema{
			{FeatureGroupLabel: "fg", Features: []*pb.Feature{{Label: "f1", ColumnIdx: 0}}},
		},
		Rows: []*pb.DecodedRow{{Keys: []string{"k1"}, Columns: []string{"v1"}}},
	}
}

// rpcCall describes one way of invoking the client so that every test below runs
// against both the raw and the decoded retrieve paths.
type rpcCall struct {
	name   string
	method string
	setup  func(m *mockRetrieveServiceClient, capture func(ctx context.Context))
	invoke func(c *ClientV1, ctx context.Context) error
}

func rpcCalls() []rpcCall {
	return []rpcCall{
		{
			name:   "RetrieveFeatures",
			method: "RetrieveFeatures",
			setup: func(m *mockRetrieveServiceClient, capture func(ctx context.Context)) {
				m.On("RetrieveFeatures", mock.Anything, mock.Anything, mock.Anything).
					Run(func(args mock.Arguments) { capture(args.Get(0).(context.Context)) }).
					Return(okRetrieveResult(), nil)
			},
			invoke: func(c *ClientV1, ctx context.Context) error {
				_, err := c.RetrieveFeatures(ctx, ctxTestQuery())
				return err
			},
		},
		{
			name:   "RetrieveDecodedFeatures",
			method: "RetrieveDecodedResult",
			setup: func(m *mockRetrieveServiceClient, capture func(ctx context.Context)) {
				m.On("RetrieveDecodedResult", mock.Anything, mock.Anything, mock.Anything).
					Run(func(args mock.Arguments) { capture(args.Get(0).(context.Context)) }).
					Return(okDecodedResult(), nil)
			},
			invoke: func(c *ClientV1, ctx context.Context) error {
				_, err := c.RetrieveDecodedFeatures(ctx, ctxTestQuery())
				return err
			},
		},
	}
}

func TestRetrieve_PropagatesCallerContextValues(t *testing.T) {
	for _, call := range rpcCalls() {
		t.Run(call.name, func(t *testing.T) {
			m := &mockRetrieveServiceClient{}
			var got context.Context
			call.setup(m, func(ctx context.Context) { got = ctx })

			ctx := context.WithValue(context.Background(), ctxKey{}, "trace-123")
			assert.NoError(t, call.invoke(newCtxTestClient(m, 1000), ctx))

			if assert.NotNil(t, got, "server was not called") {
				assert.Equal(t, "trace-123", got.Value(ctxKey{}), "values on the caller's context must reach the gRPC call")
			}
		})
	}
}

func TestRetrieve_HonoursCallerCancellation(t *testing.T) {
	for _, call := range rpcCalls() {
		t.Run(call.name, func(t *testing.T) {
			m := &mockRetrieveServiceClient{}
			// Behave like a real gRPC client: fail when the context is already done.
			failIfCancelled := func(ctx context.Context) {
				if ctx.Err() != nil {
					panic(ctx.Err()) // converted to an error response by the client's recover
				}
			}
			call.setup(m, failIfCancelled)

			ctx, cancel := context.WithCancel(context.Background())
			cancel()

			assert.Error(t, call.invoke(newCtxTestClient(m, 1000), ctx), "a cancelled caller context must fail the call")
		})
	}
}

func TestRetrieve_AppliesConfiguredDeadlineWhenCallerHasNone(t *testing.T) {
	for _, call := range rpcCalls() {
		t.Run(call.name, func(t *testing.T) {
			m := &mockRetrieveServiceClient{}
			var got context.Context
			call.setup(m, func(ctx context.Context) { got = ctx })

			start := time.Now()
			assert.NoError(t, call.invoke(newCtxTestClient(m, 1000), context.Background()))

			if assert.NotNil(t, got) {
				dl, ok := got.Deadline()
				if assert.True(t, ok, "configured DeadLine must be applied when the caller context has no deadline") {
					assert.WithinDuration(t, start.Add(time.Second), dl, 500*time.Millisecond)
				}
			}
		})
	}
}

func TestRetrieve_KeepsCallerDeadline(t *testing.T) {
	for _, call := range rpcCalls() {
		t.Run(call.name, func(t *testing.T) {
			m := &mockRetrieveServiceClient{}
			var got context.Context
			call.setup(m, func(ctx context.Context) { got = ctx })

			callerDeadline := time.Now().Add(50 * time.Millisecond)
			ctx, cancel := context.WithDeadline(context.Background(), callerDeadline)
			defer cancel()
			assert.NoError(t, call.invoke(newCtxTestClient(m, 60000), ctx))

			if assert.NotNil(t, got) {
				dl, ok := got.Deadline()
				assert.True(t, ok)
				assert.True(t, dl.Equal(callerDeadline), "caller deadline %v must win over the configured one, got %v", callerDeadline, dl)
			}
		})
	}
}

func TestRetrieve_NoDeadlineWhenNoneConfigured(t *testing.T) {
	for _, call := range rpcCalls() {
		t.Run(call.name, func(t *testing.T) {
			m := &mockRetrieveServiceClient{}
			var got context.Context
			call.setup(m, func(ctx context.Context) { got = ctx })

			assert.NoError(t, call.invoke(newCtxTestClient(m, 0), context.Background()))

			if assert.NotNil(t, got) {
				_, ok := got.Deadline()
				assert.False(t, ok, "DeadLine=0 means unset and must not produce an already-expired deadline")
				assert.NoError(t, got.Err())
			}
		})
	}
}

func TestRetrieve_NilContextAndAuthMetadata(t *testing.T) {
	for _, call := range rpcCalls() {
		t.Run(call.name, func(t *testing.T) {
			m := &mockRetrieveServiceClient{}
			var got context.Context
			call.setup(m, func(ctx context.Context) { got = ctx })

			// A nil context is explicitly supported by the client.
			assert.NoError(t, call.invoke(newCtxTestClient(m, 1000), nil)) //nolint:staticcheck

			if assert.NotNil(t, got) {
				md, ok := metadata.FromOutgoingContext(got)
				if assert.True(t, ok) {
					assert.Equal(t, []string{"test-caller"}, md.Get(headerCallerID))
					assert.Equal(t, []string{"test-token"}, md.Get(headerCallerToken))
				}
			}
		})
	}
}
