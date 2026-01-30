#include "operators/matmul.h"
#include "utils/operator_utils.h"

namespace infini {

MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                     bool transB)
    : OperatorObj(OpType::MatMul, TensorVec{A, B}, {C}),
      transA(transA), transB(transB) {
    IT_ASSERT(checkValid(graph));
}

string MatmulObj::toString() const {
    std::ostringstream os;
    os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B]")
       << ",A=" << inputs[0]->getGuid()
       << ",B=" << inputs[1]->getGuid() << ",C=" << outputs[0]->getGuid()
       << ",mnk=[" << m << "," << n << "," << k << "])";
    return os.str();
}

optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs) {
    // =================================== 作业 ===================================
    // TODO：返回经过 matmul 操作后的 shape
    // REF: https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm
    // =================================== 作业 ===================================
    auto A = inputs[0];
    auto B = inputs[1];
    auto shapeA = A->getDims();
    auto shapeB = B->getDims();
    int rankA = shapeA.size();
    int rankB = shapeB.size();
    IT_ASSERT(rankA >= 2);
    IT_ASSERT(rankB >= 2);
    int m = transA ? shapeA[rankA - 1] : shapeA[rankA - 2];
    int k = transA ? shapeA[rankA - 2] : shapeA[rankA - 1];
    int n = transB ? shapeB[rankB - 2] : shapeB[rankB - 1];
    int k_check = transB ? shapeB[rankB - 1] : shapeB[rankB - 2];
    IT_ASSERT(k == k_check);

    this->m = m;
    this->n = n;
    this->k = k;
    Shape ans = infer_broadcast(Shape(shapeA.begin(), shapeA.end() - 2),
                                Shape(shapeB.begin(), shapeB.end() - 2));
    ans.push_back(m);
    ans.push_back(n);
    return {{ans}};
}

} // namespace infini