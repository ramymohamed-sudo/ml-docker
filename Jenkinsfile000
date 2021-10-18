

// // # https://www.jenkins.io/doc/book/pipeline/docker/

// node {
//     checkout scm

//     // def customImage = docker.build("my-image:${env.BUILD_ID}", "./")    // pass directory
//     def customImage = docker.build("my-image:${env.BUILD_ID}", "./")    // --volume $HOME/workspace/ml-docker:/var/lib/python 

//     customImage.inside {
//         sh 'ls'
//         sh 'python3 train.py'
//     }
// }



// node {
//     checkout scm

//     // def customImage = docker.build("my-image:${env.BUILD_ID}", "./")    // pass directory
//     def customImage = docker.build("my-image:${env.BUILD_ID}", "./")    // --volume $HOME/workspace/ml-docker:/var/lib/python

//     customImage.inside {
//         sh 'ls'
//         sh 'python3 train.py'
//     }

//     docker.withRegistry('https://index.docker.io/v1/', 'dockerhub'){
//         customImage.push()

//     }

    
// }


node {
    checkout scm

    docker.withRegistry('https://index.docker.io/v1/', 'dockerhub') {

        def customImage = docker.build("my-image:${env.BUILD_ID}", "./")

        /* Push the container to the custom Registry */
        customImage.push()
    }
}



        
