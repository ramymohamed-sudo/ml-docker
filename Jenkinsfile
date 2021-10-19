
node{

    def commit_id 
    def customImage

    stage('Preparation'){
        checkout scm
        sh 'git rev-parse --short HEAD > .git/commit-id'  
        commit_id = readFile('.git/commit-id').trim()
    }


    stage('ml-container-test'){
        
//        def myTestContainer = docker.image('jupyter/scipy-notebook')
//         myTestContainer.pull()
//         myTestContainer.inside{
//              sh 'pip install joblib'
//              sh 'python3 train.py'
//         }
        // customImage = docker.build("my-image:${env.BUILD_ID}", "./")    // from Dockerfile in "./"
        customImage = docker.build("ramyrr/machinelearning:${commit_id}", "./")
        customImage.inside {
        sh 'ls'
        sh 'python3 train.py'
    }
       
    }
    

    stage('docker build/push'){
        docker.withRegistry('https://index.docker.io/v1/', 'dockerhub'){
            // def app = docker.build("ramyrr/machinelearning:${commit_id}", '.').push()
            customImage.push()
        }
    }    

    
}
    
