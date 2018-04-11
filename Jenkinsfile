def pythons = ["2.7.14", "3.6.4"]

def builders = [:]

for (x in pythons) {
  def python = x

  builders[python] = {
    node(python) {
      stages {
        stage('SCM') {
          sh 'python --version'
        }
        stage('Build') {
          sh 'ls -al'
        }
      }
    }
}

parallel builders
