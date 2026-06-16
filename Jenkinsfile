pipeline {
  agent any
  environment {
    AWS_REGION      = 'us-east-1'
    ECR_REGISTRY    = '492094933467.dkr.ecr.us-east-1.amazonaws.com'
    ECR_REPO        = 'ml-platform'
    EKS_CLUSTER     = 'ml-platform-dev-cluster'
    IMAGE_TAG       = "${env.BUILD_NUMBER}"
  }
  stages {
    stage('Checkout') {
      steps { checkout scm }
    }
    stage('Build Docker Image') {
      steps {
        sh 'docker build -t $ECR_REGISTRY/$ECR_REPO:$IMAGE_TAG .'
      }
    }
    stage('Push to ECR') {
      steps {
        sh '''
          aws ecr get-login-password --region $AWS_REGION | \
          docker login --username AWS --password-stdin $ECR_REGISTRY
          docker push $ECR_REGISTRY/$ECR_REPO:$IMAGE_TAG
        '''
      }
    }
    stage('Deploy to EKS') {
      steps {
        sh '''
          aws eks update-kubeconfig --region $AWS_REGION --name $EKS_CLUSTER
          kubectl set image deployment/ml-serving \
            ml-serving=$ECR_REGISTRY/$ECR_REPO:$IMAGE_TAG \
            -n ml-platform
          kubectl rollout status deployment/ml-serving -n ml-platform
        '''
      }
    }
  }
  post {
    success { echo 'Deployment successful!' }
    failure { echo 'Pipeline failed — check logs above' }
  }
}
