# WP-078: Enzyme's reverse mode through `copyto!` between CUDA arrays (does it compile?)
using CUDACore, Enzyme
f!(a, b) = (copyto!(a, b); nothing)
a = CUDACore.zeros(Float32, 16); b = CUDACore.ones(Float32, 16)
da = CUDACore.ones(Float32, 16); db = CUDACore.zeros(Float32, 16)
Enzyme.autodiff(set_runtime_activity(Reverse), f!, Const, Duplicated(a, da), Duplicated(b, db))
println("copyto! reverse OK: db = ", Array(db)[1:3])
