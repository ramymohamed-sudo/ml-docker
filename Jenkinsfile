


node{

    def commit_id 

    stage('Preparation'){
        checkout scm
        sh 'git rev-parse --short HEAD > .git/commit-id'  
        commit_id = readFile('.git/commit-id').trim()
    }

    stage('docker build/push'){
        docker.withRegistry('https://index.docker.io/v1/', 'dockerhub'){
            def app = docker.build("ramyrr/machinelearning:${commit_id}", '.').push()
        }
    }    


    // stage('run-ml-container'){
    //     def myTestContainer = docker.image('ramyrr/machinelearning:latest')
    //     myTestContainer.pull()
    //     myTestContainer.inside{
    //          sh 'python3 train.py'

    //     }

    // }

