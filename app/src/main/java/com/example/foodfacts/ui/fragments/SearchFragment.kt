package com.example.foodfacts.ui.fragments

import android.os.Bundle
import android.text.Editable
import android.text.TextWatcher
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.view.inputmethod.EditorInfo
import android.widget.Toast
import androidx.fragment.app.Fragment
import androidx.fragment.app.viewModels
import androidx.navigation.fragment.findNavController
import androidx.recyclerview.widget.LinearLayoutManager
import com.example.foodfacts.R
import com.example.foodfacts.databinding.FragmentSearchBinding
import com.example.foodfacts.ui.adapters.ProductAdapter
import com.example.foodfacts.ui.viewmodels.SearchUiState
import com.example.foodfacts.ui.viewmodels.SearchViewModel

class SearchFragment : Fragment() {

    private var _binding: FragmentSearchBinding? = null
    private val binding get() = _binding!!

    private val viewModel: SearchViewModel by viewModels()
    private lateinit var adapter: ProductAdapter

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentSearchBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        setupRecyclerView()
        setupSearchInput()
        observeViewModel()

        val ocrQuery = arguments?.getString("ocrQuery")
        if (!ocrQuery.isNullOrEmpty()) {
            binding.etSearchQuery.setText(ocrQuery)
            viewModel.search(ocrQuery, debounceMs = 0)
        }
    }

    private fun setupRecyclerView() {
        adapter = ProductAdapter(
            onItemClick = { product ->
                viewModel.recordManualSelectEvent(product)
                val bundle = Bundle().apply {
                    putString("barcode", product.barcode)
                }
                findNavController().navigate(R.id.action_searchFragment_to_productDetailFragment, bundle)
            }
        )
        binding.rvSearchResults.layoutManager = LinearLayoutManager(requireContext())
        binding.rvSearchResults.adapter = adapter
    }

    private fun setupSearchInput() {
        binding.etSearchQuery.addTextChangedListener(object : TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {}
            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {
                val query = s?.toString() ?: ""
                viewModel.search(query, debounceMs = 500)
            }
            override fun afterTextChanged(s: Editable?) {}
        })

        binding.etSearchQuery.setOnEditorActionListener { _, actionId, _ ->
            if (actionId == EditorInfo.IME_ACTION_SEARCH) {
                val query = binding.etSearchQuery.text?.toString() ?: ""
                viewModel.search(query, debounceMs = 0)
                true
            } else {
                false
            }
        }
    }

    private fun observeViewModel() {
        viewModel.uiState.observe(viewLifecycleOwner) { state ->
            when (state) {
                is SearchUiState.Idle -> {
                    binding.progressBar.visibility = View.GONE
                    binding.rvSearchResults.visibility = View.GONE
                    binding.tvStateMessage.visibility = View.VISIBLE
                    binding.tvStateMessage.text = "Type product name or barcode to search Open Food Facts database"
                }
                is SearchUiState.Loading -> {
                    binding.progressBar.visibility = View.VISIBLE
                    binding.rvSearchResults.visibility = View.GONE
                    binding.tvStateMessage.visibility = View.GONE
                }
                is SearchUiState.Success -> {
                    binding.progressBar.visibility = View.GONE
                    binding.tvStateMessage.visibility = View.GONE
                    binding.rvSearchResults.visibility = View.VISIBLE
                    adapter.submitList(state.products)
                }
                is SearchUiState.Empty -> {
                    binding.progressBar.visibility = View.GONE
                    binding.rvSearchResults.visibility = View.GONE
                    binding.tvStateMessage.visibility = View.VISIBLE
                    binding.tvStateMessage.text = "No products found matching your search term."
                }
                is SearchUiState.Error -> {
                    binding.progressBar.visibility = View.GONE
                    binding.rvSearchResults.visibility = View.GONE
                    binding.tvStateMessage.visibility = View.VISIBLE
                    binding.tvStateMessage.text = "Search Error: ${state.message}"
                    Toast.makeText(requireContext(), state.message, Toast.LENGTH_SHORT).show()
                }
            }
        }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}
