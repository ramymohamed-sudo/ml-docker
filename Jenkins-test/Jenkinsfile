

// pipeline {

//     agent { docker { image 'python:3.5.1' } }
    

//     stages {
        
//         stage('build') {
//             steps {
//                 sh 'python --version'
//             }
//         }
        
//     }

// }

node{

    def commit_id 

    stage('Preparation'){
        checkout scm
        sh 'git rev-parse --short HEAD > .git/commit-id'  
        commit_id = readFile('.git/commit-id').trim()
    }

    stage('test'){
        def myTestContainer = docker.image('python:latest')
        myTestContainer.pull()
        myTestContainer.inside {
        sh 'python3 --version'
        // sh 'python3 train-rul.py'
        }
        
        // nodejs(nodeJSInstallationName: 'nodejs'){
        //     sh 'npm install --only-dev'
        //     // sh 'npm test'
        // }
    }

    stage('docker build/push'){
        docker.withRegistry('https://index.docker.io/v1/', 'dockerhub'){
            def app = docker.build("ramyrr/docker-node-js-demo:${commit_id}", '.').push()
        }
    }    

}