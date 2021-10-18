
node{

    def commit_id 

    stage('Preparation'){
        checkout scm
        sh 'git rev-parse --short HEAD > .git/commit-id'  
        commit_id = readFile('.git/commit-id').trim()
    }

    // def customImage = docker.build("my-image:${env.BUILD_ID}", "./")    // pass directory
    def customImage = docker.build("my-image:${env.BUILD_ID}", "./")    // --volume $HOME/workspace/ml-docker:/var/lib/python

    stage('ml-container-test'){
        
//        def myTestContainer = docker.image('jupyter/scipy-notebook')
//         myTestContainer.pull()
//         myTestContainer.inside{
//              sh 'pip install joblib'
//              sh 'python3 train.py'
//         }
        def customImage = docker.build("my-image:${env.BUILD_ID}", "./")    
        customImage.inside {
        sh 'ls'
        sh 'python3 train.py'
    }
<<<<<<< HEAD

    docker.withRegistry('https://index.docker.io/v1/', 'dockerhub'){
        customImage.push()

    }

    
}




        
=======
        

    }
    

    stage('docker build/push'){
        docker.withRegistry('https://index.docker.io/v1/', 'dockerhub'){
            def app = docker.build("ramyrr/machinelearning:${commit_id}", '.').push()
        }
    }    

    
}
    
// to be completed below https://www.youtube.com/watch?v=gdbA3vR2eDs
//     stage('run-container-on-analytical-server'){
//         def myTestContainer = docker.image('jupyter/scipy-notebook')
//         myTestContainer.pull()
//         myTestContainer.inside{
//              sh 'docker run -p 5667:5667 -d -name my-ml-app ramyrr/machinelearning:${commit_id}'

//         }

>>>>>>> ce7d8e1e5fc40c68aa06faaf57cbe68b97ff0d4f
