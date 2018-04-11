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
                sh 'python -m venv venv'
            }
            stage("Build") {
                try { 
                    sh """
                        . venv/bin/activate
                        make install-dev
                    """
                } catch (e) { 
                    rocketSend channel: '#km3pipe', message: "Build Failed - ${env.JOB_NAME} ${env.BUILD_NUMBER} (<${env.BUILD_URL}|Open>)"
                    throw e
                }
            }
            stage('Test') {
                try { 
                    sh """
                        . venv/bin/activate
                        make test
                    """
                    junit 'junit.xml'
                } catch (e) { 
                    rocketSend channel: '#km3pipe', message: "Test Suite Failed - ${env.JOB_NAME} ${env.BUILD_NUMBER} (<${env.BUILD_URL}|Open>)"
                    throw e
                }
            }
            stage('Docs') {
                try { 
                    sh """
                        . venv/bin/activate
                        make doc-dependencies
                        cd docs
                        make html
                    """
                } catch (e) { 
                    rocketSend channel: '#km3pipe', message: "Building the Docs Failed - ${env.JOB_NAME} ${env.BUILD_NUMBER} (<${env.BUILD_URL}|Open>)"
                    throw e
                }
            }

        }
    }
}
