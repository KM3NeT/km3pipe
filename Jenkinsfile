def python_versions = ["2.7.14", "3.6.4"]

def builders = [:]

for (x in python_version) {
  def python_version = x

  builders[python_version] = {
    node {
      stages {
        stage('SCM') {
          docker.image("python:${python_version}").inside {
            sh 'python --version'
          }
        }
        stage('Build') {
          docker.image("python:${python_version}").inside {
            sh 'python --version'
            sh 'ls -al'
          }
        }
      }
    }
  }
}

parallel builders
