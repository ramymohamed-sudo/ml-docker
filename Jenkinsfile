


// node{

//     def commit_id 

//     stage('Preparation'){
//         checkout scm
//         sh 'git rev-parse --short HEAD > .git/commit-id'  
//         commit_id = readFile('.git/commit-id').trim()
//     }

//     // stage('docker build/push'){
//     //     docker.withRegistry('https://index.docker.io/v1/', 'dockerhub'){
//     //         def app = docker.build("ramyrr/machinelearning:${commit_id}", '.').push()
//     //     }
//     // }    


//     stage('run-ml-container'){
//         def myTestContainer = docker.image('ramyrr/machinelearning:c6070a4')
//         myTestContainer.pull()
//         myTestContainer.inside{
//              sh 'python3 train.py'

//         }

//     }

// }

def commit_id

pipeline {
    agent {
        docker {
            image 'jupyter/scipy-notebook'        
            args '-v $HOME/workspace/ml-docker:/var/lib/python'
            // args '-v /var/lib/python:/var/lib/python'
            
        }
    }

    stages {
    stage('Preparation'){
        steps {
        // checkout scm
        script {
        sh 'git rev-parse --short HEAD > .git/commit-id' 
        commit_id = readFile('.git/commit-id').trim()
        }

        }
    }

    stage('Build') {
            steps {
                sh 'pip install joblib'
                sh 'python3 train.py'
            }
        }

    // stage('push'){
    //     steps {
    //     docker.withRegistry('https://index.docker.io/v1/', 'dockerhub'){
    //         def app = docker.build("ramyrr/machinelearning:${commit_id}", '.').push()
    //     }
    //     }
    // } 

    }

  

}
