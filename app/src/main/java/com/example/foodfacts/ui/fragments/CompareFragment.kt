package com.example.foodfacts.ui.fragments

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.fragment.app.Fragment
import androidx.fragment.app.viewModels
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import androidx.recyclerview.widget.LinearLayoutManager
import com.example.foodfacts.data.model.ProductDomainModel
import com.example.foodfacts.databinding.FragmentCompareBinding
import com.example.foodfacts.ui.adapters.MetricAdapter
import com.example.foodfacts.ui.viewmodels.CompareUiState
import com.example.foodfacts.ui.viewmodels.CompareViewModel
import kotlinx.coroutines.launch

class CompareFragment : Fragment() {

    private var _binding: FragmentCompareBinding? = null
    private val binding get() = _binding!!

    private val viewModel: CompareViewModel by viewModels()
    private lateinit var adapter: MetricAdapter

    private var cachedHistory: List<ProductDomainModel> = emptyList()

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentCompareBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        setupRecyclerView()
        setupListeners()
        observeViewModel()

        val productABarcode = arguments?.getString("productABarcode")
        if (!productABarcode.isNullOrEmpty()) {
            viewModel.loadAndSetProductA(productABarcode)
        }
    }

    private fun setupRecyclerView() {
        adapter = MetricAdapter()
        binding.rvMetrics.layoutManager = LinearLayoutManager(requireContext())
        binding.rvMetrics.adapter = adapter
    }

    private fun setupListeners() {
        binding.cardProductA.setOnClickListener {
            showProductPickerDialog(isProductA = true)
        }
        binding.cardProductB.setOnClickListener {
            showProductPickerDialog(isProductA = false)
        }
    }

    private fun showProductPickerDialog(isProductA: Boolean) {
        if (cachedHistory.isEmpty()) {
            Toast.makeText(requireContext(), "No products in history. Scan or search for products first!", Toast.LENGTH_LONG).show()
            return
        }

        val options = cachedHistory.map { "${it.name} (${it.brand})" }.toTypedArray()
        AlertDialog.Builder(requireContext())
            .setTitle(if (isProductA) "Select Product 1" else "Select Product 2")
            .setItems(options) { _, which ->
                val selectedProduct = cachedHistory[which]
                if (isProductA) {
                    viewModel.setProductA(selectedProduct)
                } else {
                    viewModel.setProductB(selectedProduct)
                }
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun observeViewModel() {
        viewLifecycleOwner.lifecycleScope.launch {
            viewLifecycleOwner.repeatOnLifecycle(Lifecycle.State.STARTED) {
                viewModel.historyProducts.collect { products ->
                    cachedHistory = products
                }
            }
        }

        viewModel.productA.observe(viewLifecycleOwner) { product ->
            if (product != null) {
                binding.tvProductAName.text = product.name
                binding.tvProductABrand.text = product.brand
            } else {
                binding.tvProductAName.text = "Select Product 1"
                binding.tvProductABrand.text = "Tap to choose"
            }
        }

        viewModel.productB.observe(viewLifecycleOwner) { product ->
            if (product != null) {
                binding.tvProductBName.text = product.name
                binding.tvProductBBrand.text = product.brand
            } else {
                binding.tvProductBName.text = "Select Product 2"
                binding.tvProductBBrand.text = "Tap to choose"
            }
        }

        viewModel.uiState.observe(viewLifecycleOwner) { state ->
            when (state) {
                is CompareUiState.SelectionPending -> {
                    binding.tvSummary.text = "Select two products from history to evaluate comparison"
                    adapter.submitList(emptyList())
                }
                is CompareUiState.Loading -> {
                    binding.tvSummary.text = "Calculating nutritional metrics..."
                }
                is CompareUiState.Success -> {
                    binding.tvSummary.text = state.comparisonResult.summary
                    adapter.submitList(state.comparisonResult.metrics)
                }
                is CompareUiState.Error -> {
                    binding.tvSummary.text = state.message
                }
            }
        }
    }



    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}
