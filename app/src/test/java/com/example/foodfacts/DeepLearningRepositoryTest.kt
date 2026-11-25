package com.example.foodfacts

import com.example.foodfacts.data.model.Resource
import com.example.foodfacts.integration.prediction.DeepLearningRepository
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

@OptIn(ExperimentalCoroutinesApi::class)
class DeepLearningRepositoryTest {

    @Test
    fun `predictNovaGroup returns Error when text is empty`() = runTest {
        val repository = DeepLearningRepository()
        val result = repository.predictNovaGroup("")

        assertTrue(result is Resource.Error)
        assertEquals("Ingredients text is empty. Please enter or extract ingredient text.", (result as Resource.Error).message)
    }

    @Test
    fun `predictNovaGroup handles unreachable server gracefully`() = runTest {
        val repository = DeepLearningRepository(baseUrl = "http://127.0.0.1:9999/")
        val result = repository.predictNovaGroup("Sugar, Palm Oil")

        assertTrue(result is Resource.Error)
        assertTrue((result as Resource.Error).message.contains("AI prediction service is not connected"))
    }
}
