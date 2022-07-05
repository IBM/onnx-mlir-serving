# ONNX Serving

## Serving Tool Proposal

ONNX Serving is a project written with C++ to serve onnx-mlir compiled models with GRPC and other protocols. Benefiting from C++ implementation, ONNX Serving has very low latency overhead and high throughput. ONNX Servring provides dynamic batch aggregation and workers pool to fully utilize AI accelerators on the machine.

Currently there is no existing high performance open source sering solution for onnx-mlir compiled model, IBM wants to contribute an open-source project to ONNX community which can help user to deploy their onnx-mlir in production environment.

## Proposal

Contriubte ONNX Serving to https://github.com/onnx/onnx-serving

Welcome community contributions to enhance onnx-serving with broader hardware and platform support.

Questions:


## Rules for all repos and Requirements for new, contributed repos

| Rules for all repos

1. Must be owned and managed by one of the ONNX SIGs (Architecture & Infra)

2. Must be actively maintained (Qin Yue Chen, Fei Fei Li)

3. Must adopt the ONNX Code of Conduct (check)

4. Must adopt the standard ONNX license(s) (already Apache-2.0 License)

5. Must adopt the ONNX CLA bot (check)

6. Must adopt all ONNX automation (like LGTM) (check)

7. Must have CI or other automation in place for repos containing code to ensure quality (already implemented CI and utest, need to implement more test cases and add coverage scan tool)

8. | All OWNERS must be members of standing as defined by ability to vote in Steering Committee elections. (check)

Requirements for new, contributed repos

We are happy to accept contributions as repos under the ONNX organization of new projects that meet the following requirements:

1. Project is closely related to ONNX (onnx-mlir)

2. Adds value to the ONNX ecosystem (serving onnx-mlir compiled model)

3. Determined to need a new repo rather than a folder in an existing repo (no)

4. All contributors must have signed the ONNX CLA (check)

5. Licenses of dependencies must be acceptable (check)

6. Committment to maintain the repo (Qin Yue Chen, Fei Fei Li)

7. Approval of the SIG that will own the repo

8. Approval of the Steering Committee