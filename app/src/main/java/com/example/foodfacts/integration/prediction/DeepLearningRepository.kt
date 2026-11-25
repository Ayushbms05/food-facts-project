package com.example.foodfacts.integration.prediction

import com.example.foodfacts.data.model.Resource
import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import java.util.concurrent.TimeUnit

class DeepLearningRepository(
    var baseUrl: String = "http://10.0.2.2:8000/"
) {
    private fun createApi(): DeepLearningApi {
        val logging = HttpLoggingInterceptor().apply {
            level = HttpLoggingInterceptor.Level.BODY
        }
        val client = OkHttpClient.Builder()
            .addInterceptor(logging)
            .connectTimeout(5, TimeUnit.SECONDS)
            .readTimeout(5, TimeUnit.SECONDS)
            .build()

        val formattedUrl = if (baseUrl.endsWith("/")) baseUrl else "$baseUrl/"

        return Retrofit.Builder()
            .baseUrl(formattedUrl)
            .client(client)
            .addConverterFactory(GsonConverterFactory.create())
            .build()
            .create(DeepLearningApi::class.java)
    }

    suspend fun predictNovaGroup(ingredientsText: String): Resource<PredictionResponse> {
        val cleanText = ingredientsText.trim()
        if (cleanText.isEmpty()) {
            return Resource.Error("Ingredients text is empty. Please enter or extract ingredient text.")
        }
        return try {
            val api = createApi()
            val response = api.predictNovaGroup(PredictionRequest(cleanText))
            if (response.isSuccessful && response.body() != null) {
                Resource.Success(response.body()!!)
            } else {
                Resource.Error("Deep Learning API error (${response.code()}): ${response.message()}")
            }
        } catch (e: Exception) {
            Resource.Error(
                "AI prediction service is not connected ($baseUrl). Member 2's Deep Learning service is offline.",
                cause = e
            )
        }
    }
}
