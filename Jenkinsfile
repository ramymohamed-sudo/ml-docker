
node{

    def commit_id 

    stage('Preparation'){
        checkout scm
        sh 'git rev-parse --short HEAD > .git/commit-id'  
        commit_id = readFile('.git/commit-id').trim()
    }


        stage('test'){
        def myTestContainer = docker.image('node:latest')
        myTestContainer.pull()
        myTestContainer.inside{
             sh 'npm -version'

        }
        // nodejs(nodeJSInstallationName: 'nodejs'){
        //     sh 'npm install --only-dev'
        //     sh 'npm test'
        // }
    
    }


    // stage('python build') {
    //     // steps {
    //         sh 'python3 train.py'
    //     // }
    // }


    stage('docker build/push'){
        docker.withRegistry('https://index.docker.io/v1/', 'dockerhub'){
            def app = docker.build("ramyrr/docker-node-js-demo:${commit_id}", '.').push()
        }
    }    

}