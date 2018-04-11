def pythons = ["2.7.14", "3.6.4"]

def steps_scm = pythons.collectEntries {
    ["python $it": step_scm(it)]
}

def steps_build = pythons.collectEntries {
    ["python $it": step_build(it)]
}

parallel steps_scm
parallel steps_build

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
