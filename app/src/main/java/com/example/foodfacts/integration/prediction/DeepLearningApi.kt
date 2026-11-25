package com.example.foodfacts.integration.prediction

import retrofit2.Response
import retrofit2.http.Body
import retrofit2.http.POST

interface DeepLearningApi {

    @POST("api/v1/predict-nova")
    suspend fun predictNovaGroup(
        @Body request: PredictionRequest
    ): Response<PredictionResponse>
}
