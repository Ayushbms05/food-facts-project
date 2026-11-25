package com.example.foodfacts.data.model

sealed class Resource<out T> {
    data class Success<out T>(val data: T) : Resource<T>()
    data class Error<out T>(
        val message: String,
        val cause: Throwable? = null,
        val cachedData: T? = null
    ) : Resource<T>()
    data class Loading<out T>(val cachedData: T? = null) : Resource<T>()
}
