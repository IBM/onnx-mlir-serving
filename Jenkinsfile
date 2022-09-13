pipeline {
    agent any
    stages {
        stage('Build Base Image') {
            steps {
                sh "docker build -f Dockerfile.base -t onnx/aigrpc-base ."
            }
        }
        stage('Build Server Image') {
            steps {
                sh "docker build -t onnx/aigrpc-server ."
            }
        }
        stage('Build Debug Image') {
            steps {
                sh "docker build -f Dockerfile.debug -t onnx/aigrpc-debug  ."
            }
        }
        stage('utest') {
            steps {
                sh "docker run -v ${PWD}/results:/results onnx/aigrpc-server -c 'cd /workdir/aigrpc-server/cmake/build;./grpc-test --gtest_output=xml:/results/'"
            }
        }
    }
    post {
        always { 
          junit './results/**'   
        }
        success {
            echo 'This will run only if successful'
        }
        failure {
            echo 'This will run only if failed'
        }
        unstable {
            echo 'This will run only if the run was marked as unstable'
        }
    }
}