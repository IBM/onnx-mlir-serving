pipeline {
    agent any
    stages {
        stage('Build Base Image') {
            steps {
                sh "docker build -f Dockerfile.base -t onnx/aigrpc-base ."
            }
        }
        stage('Build Dev Image') {
            steps {
                sh "docker build -t onnx/aigrpc-server ."
            }
        }
        stage('utest') {
            steps {
                sh "docker run -v /results:./results onnx/aigrpc-server -c 'cd /workdir/aigrpc-server/cmake/build;./grpc-test --gtestout=xml:/results/utest-grpc.xml'"
            }
        }
    }
    post {
        always { 
          junit './results/utest-grpc.xml'   
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