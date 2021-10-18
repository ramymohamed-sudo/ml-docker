

// # https://www.jenkins.io/doc/book/pipeline/docker/

node {
    checkout scm

    // def customImage = docker.build("my-image:${env.BUILD_ID}", "./")    // pass directory
    def customImage = docker.build("my-image:${env.BUILD_ID}", "-v $HOME/workspace/ml-docker:/var/lib/python ./")    // pass directory

    customImage.inside {
        sh 'ls'
        sh 'python3 train.py'
    }
}