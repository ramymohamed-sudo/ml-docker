
node{

    def commit_id 
    def customImage

    stage('Preparation'){
        checkout scm
        sh 'git rev-parse --short HEAD > .git/commit-id'  
        commit_id = readFile('.git/commit-id').trim()
    }


    stage('Build'){

// Build from image        
//        def myTestContainer = docker.image('jupyter/scipy-notebook')
//         myTestContainer.pull()
//         myTestContainer.inside{
//              sh 'pip install joblib'
//              sh 'python3 train.py'
//         }

        // Build from Dockerfile  // from Dockerfile in "./"
        // customImage = docker.build("my-image:${env.BUILD_ID}", "./")    
        customImage = docker.build("ramyrr/machinelearning:${commit_id}", "./")
       
    }
    
    stage('Run'){
        
        customImage.inside {
        sh 'ls'
        sh 'python3 train.py'
    }

    }


    stage('Push'){
        docker.withRegistry('https://index.docker.io/v1/', 'dockerhub'){
            // def app = docker.build("ramyrr/machinelearning:${commit_id}", '.').push()
            customImage.push()

        }
    }    

    
}
    
