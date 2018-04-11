def pythons = ["2.7.14", "3.6.4"]

def steps_scm = pythons.collectEntries {
    ["python $it": step_scm(it)]
}

def steps_build = pythons.collectEntries {
    ["python $it": step_build(it)]
}

def step_scm(version) {
    return {
        docker.image("python:${version}").inside {
            checkout scm
        }
    }
}

def step_build(version) {
    return {
        docker.image("python:${version}").inside {
            sh 'python --version'
        }
    }
}

node {
    stage('SCM') {
        parallel steps_scm
    }
    stage('Build') {
        parallel steps_build
    }
}

