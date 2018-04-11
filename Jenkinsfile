def pythons = ["2.7.14", "3.6.4"]

def steps = pythons.collectEntries {
    ["python $it": job(it)]
}

parallel steps

def job(version) {
    return {
        docker.image("python:${version}").inside {
            checkout scm
            sh 'touch python${version}'
            sh 'ls -al'
            sh 'python --version'
        }
    }
}
