node {
    stage("Python 2.7.14") {
        checkout scm
        docker.image('python:2.7.14').inside {
            stage("Prepare") {
                sh 'python --version'
            }
            stage("Build") {
                sh 'ls -al'
            }
        }
    }
    stage("Python 3.6.4") {
        checkout scm
        docker.image('python:3.6.4').inside {
            stage("Prepare") {
                sh 'python --version'
            }
            stage("Build") {
                sh 'ls -al'
            }
        }
    }
}
