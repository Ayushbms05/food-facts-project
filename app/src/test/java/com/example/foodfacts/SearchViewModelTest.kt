package com.example.foodfacts

import com.example.foodfacts.data.api.OpenFoodFactsApi
import com.example.foodfacts.data.api.model.ApiProduct
import com.example.foodfacts.data.api.model.SearchResponse
import com.example.foodfacts.data.local.ProductDao
import com.example.foodfacts.data.local.ScanEventDao
import com.example.foodfacts.data.model.Resource
import com.example.foodfacts.data.repository.ProductRepository
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.mockito.kotlin.mock
import org.mockito.kotlin.whenever
import retrofit2.Response

@OptIn(ExperimentalCoroutinesApi::class)
class SearchViewModelTest {

    private lateinit var productDao: ProductDao
    private lateinit var scanEventDao: ScanEventDao
    private lateinit var api: OpenFoodFactsApi
    private lateinit var repository: ProductRepository

    @Before
    fun setUp() {
        productDao = mock()
        scanEventDao = mock()
        api = mock()
        repository = ProductRepository(productDao, scanEventDao, api)
    }

    @Test
    fun `searchProducts returns list of products on successful API query`() = runTest {
        val query = "Milk"
        val apiProduct = ApiProduct(
            code = "100",
            productName = "Whole Milk",
            brands = "Dairy",
            nutriscoreGrade = "A"
        )
        val response = SearchResponse(count = 1, products = listOf(apiProduct))

        whenever(api.searchProducts(searchTerms = query)).thenReturn(Response.success(response))

        val result = repository.searchProducts(query)

        assertTrue(result is Resource.Success)
        val list = (result as Resource.Success).data
        assertEquals(1, list.size)
        assertEquals("Whole Milk", list[0].name)
    }

    @Test
    fun `searchProducts returns Error when server returns HTTP error`() = runTest {
        val query = "Unknown"
        whenever(api.searchProducts(searchTerms = query)).thenReturn(Response.error(500, okhttp3.ResponseBody.create(null, "Server Error")))

        val result = repository.searchProducts(query)

        assertTrue(result is Resource.Error)
        val error = result as Resource.Error
        assertTrue(error.message.contains("500"))
    }
}
