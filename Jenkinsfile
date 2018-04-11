def pythons = ["2.7.14", "3.6.4"]

def steps_scm = pythons.collectEntries {
    ["python $it": step_scm(it)]
}

def steps_build = pythons.collectEntries {
    ["python $it": step_build(it)]
}

def steps_prepare = pythons.collectEntries {
    ["python $it": step_prepare(it)]
}

def step_scm(version) {
    return {
        docker.image("python:${version}").inside {
            checkout scm
        }
    }
}

def step_prepare(version) {
    return {
        docker.image("python:${version}").inside {
            sh 'python -m venv venv'
        }
    }
}

def step_build(version) {
    return {
        docker.image("python:${version}").inside {
            script {
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
        }
    }
}

node {
    stage('SCM') {
        parallel steps_scm
    }
    stage('Prepare') {
        parallel steps_prepare
    }
    stage('Build') {
        parallel steps_build
    }
}

