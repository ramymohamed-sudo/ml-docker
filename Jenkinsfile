


pipeline {
    agent {
        docker {
            image 'python:latest'
            args '-v /var/lib/python:/var/lib/python'
            
        }
    }
    stages {
        stage('Build') {
            steps {
                sh 'python3 file.py'
            }
        }
    }
}
