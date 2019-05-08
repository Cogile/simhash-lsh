package simhashlsh

import (
	"fmt"
	"log"
	"math/rand"
	"strconv"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestSimpleQuery(t *testing.T) {
	lsh := NewCosineLsh(8, 1, 6)
	lsh.Insert([]float64{1, 2, 3, 4, 5, 6, 7, 8}, "1")
	lsh.Insert([]float64{2, 3, 4, 5, 6, 7, 8, 9}, "2")
	lsh.Insert([]float64{10, 12, 99, 1, 5, 31, 2, 3}, "3")
	res := lsh.Query([]float64{1, 2, 3, 4, 5, 6, 7, 7}, -1)

	require.Len(t, res, 2)

	assert.Equal(t, float64(1), res[0].Distance)
	assert.Equal(t, float64(11), res[1].Distance)
}

func TestSimpleQueryWithMaxResult(t *testing.T) {
	lsh := NewCosineLsh(8, 1, 6)
	for i := 0; i < 10; i++ {
		lsh.Insert([]float64{1, 2, 3, 4, 5, 6, 7, 8}, strconv.Itoa(i))
	}
	res := lsh.Query([]float64{1, 2, 3, 4, 5, 6, 7, 7}, -1)
	require.Len(t, res, 10)

	res = lsh.Query([]float64{1, 2, 3, 4, 5, 6, 7, 7}, 5)
	require.Len(t, res, 5)
}

func Test_CosineLshQuery(t *testing.T) {
	ls := []int{20, 5, 10, 25, 4}
	ks := []int{5, 20, 10, 4, 25}
	embs := randomEmbeddings(1000, 300, 1.0)
	for j := range ls {
		var avg float64
		clsh := NewCosineLsh(300, ls[j], ks[j])
		insertedEmbs := make([]string, 1000)
		for i, e := range embs {
			clsh.Insert(e, strconv.Itoa(i))
			insertedEmbs[i] = strconv.Itoa(i)
		}
		// Use the inserted embeddings as queries, and
		// verify that we can get back each query itself
		log.Printf("avg number of returned results with k=%d and l=%d for each query (out of %d indexed items): ", ks[j], ls[j], len(embs))
		for i, key := range insertedEmbs {
			results := clsh.Query(embs[i], -1)
			avg += float64(len(results))
			found := false
			for _, foundKey := range results {
				if foundKey.ExtraData == key {
					found = true
				}
			}
			if !found {
				t.Error("Query fail")
			}
		}
		log.Printf(" %f", avg/float64(len(embs)))
	}
}

func Test_CosineLshQueryShouldBeSortedByL2(t *testing.T) {
	ls := []int{20, 5, 10, 25, 4}
	ks := []int{5, 20, 10, 4, 25}
	embs := randomEmbeddings(1000, 300, 1.0)
	for j := range ls {
		var avg float64
		clsh := NewCosineLsh(300, ls[j], ks[j])
		insertedEmbs := make([]string, 1000)
		for i, e := range embs {
			clsh.Insert(e, strconv.Itoa(i))
			insertedEmbs[i] = strconv.Itoa(i)
		}
		// Use the inserted embeddings as queries, and
		// verify that we can get back each query itself
		log.Printf("avg number of returned results with k=%d and l=%d for each query (out of %d indexed items): ", ks[j], ls[j], len(embs))
		for i := range insertedEmbs {
			results := clsh.Query(embs[i], -1)
			var lastDistance float64
			avg += float64(len(results))
			//
			//if results[0].ID != key {
			//	t.Fatal("first key should be query vector")
			//}

			for _, res := range results {
				if len(res.Vector) == 0 {
					panic(fmt.Errorf("nil vector %+v", res))
				}
			}
			for _, foundKey := range results {

				dist := euclideanDistSquare(embs[i], foundKey.Vector)

				if dist < lastDistance {
					t.Fatal("query should sort vectors by distance")
				}
				lastDistance = dist
			}
		}
		log.Printf(" %f", avg/float64(len(embs)))
	}
}

func BenchmarkCosineLshQueryNaive(b *testing.B) {
	b.StopTimer()
	lshSize := 90000
	vecSize := 512
	lsh := NewCosineLsh(vecSize, 5, 10)
	embs := randomEmbeddings(lshSize, vecSize, 1.0)
	for i, e := range embs {
		lsh.Insert(e, strconv.Itoa(i))
	}
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		lsh.Query(embs[0], -1)
	}
}

func randomEmbeddings(n, dim int, max float64) [][]float64 {
	random := rand.New(rand.NewSource(1))
	embs := make([][]float64, n)
	for i := 0; i < n; i++ {
		embs[i] = make([]float64, dim)
		for d := 0; d < dim; d++ {
			embs[i][d] = random.Float64() * max
		}
	}
	return embs
}
