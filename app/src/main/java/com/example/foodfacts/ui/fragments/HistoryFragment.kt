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
import androidx.navigation.fragment.findNavController
import androidx.recyclerview.widget.LinearLayoutManager
import com.example.foodfacts.R
import com.example.foodfacts.databinding.FragmentHistoryBinding
import com.example.foodfacts.ui.adapters.ProductAdapter
import com.example.foodfacts.ui.adapters.ScanEventAdapter
import com.example.foodfacts.ui.viewmodels.HistoryViewModel
import com.google.android.material.tabs.TabLayout
import kotlinx.coroutines.launch

class HistoryFragment : Fragment() {

    private var _binding: FragmentHistoryBinding? = null
    private val binding get() = _binding!!

    private val viewModel: HistoryViewModel by viewModels()

    private lateinit var productAdapter: ProductAdapter
    private lateinit var scanEventAdapter: ScanEventAdapter

    private var currentTabPosition = 0

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentHistoryBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        setupAdapters()
        setupListeners()
        observeViewModel()
    }

    private fun setupAdapters() {
        productAdapter = ProductAdapter(
            onItemClick = { product ->
                val bundle = Bundle().apply { putString("barcode", product.barcode) }
                findNavController().navigate(R.id.action_historyFragment_to_productDetailFragment, bundle)
            },
            onItemLongClick = { product ->
                showDeleteProductDialog(product.barcode, product.name)
            }
        )

        scanEventAdapter = ScanEventAdapter(
            onItemClick = { event ->
                val bundle = Bundle().apply { putString("barcode", event.barcode) }
                findNavController().navigate(R.id.action_historyFragment_to_productDetailFragment, bundle)
            }
        )

        binding.rvHistory.layoutManager = LinearLayoutManager(requireContext())
        binding.rvHistory.adapter = productAdapter
    }

    private fun setupListeners() {
        binding.tabLayoutHistory.addOnTabSelectedListener(object : TabLayout.OnTabSelectedListener {
            override fun onTabSelected(tab: TabLayout.Tab?) {
                currentTabPosition = tab?.position ?: 0
                if (currentTabPosition == 0) {
                    binding.rvHistory.adapter = productAdapter
                    updateEmptyState(productAdapter.itemCount == 0, "No cached products in history yet")
                } else {
                    binding.rvHistory.adapter = scanEventAdapter
                    updateEmptyState(scanEventAdapter.itemCount == 0, "No scan events recorded yet")
                }
            }
            override fun onTabUnselected(tab: TabLayout.Tab?) {}
            override fun onTabReselected(tab: TabLayout.Tab?) {}
        })

        binding.btnClearHistory.setOnClickListener {
            AlertDialog.Builder(requireContext())
                .setTitle("Clear History")
                .setMessage("Are you sure you want to clear all history and scan logs?")
                .setPositiveButton("Clear") { _, _ ->
                    viewModel.clearAllHistory()
                    Toast.makeText(requireContext(), "History cleared", Toast.LENGTH_SHORT).show()
                }
                .setNegativeButton("Cancel", null)
                .show()
        }
    }

    private fun showDeleteProductDialog(barcode: String, name: String) {
        AlertDialog.Builder(requireContext())
            .setTitle("Remove Product")
            .setMessage("Remove '$name' from history?")
            .setPositiveButton("Remove") { _, _ ->
                viewModel.deleteHistoryProduct(barcode)
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun observeViewModel() {
        viewLifecycleOwner.lifecycleScope.launch {
            viewLifecycleOwner.repeatOnLifecycle(Lifecycle.State.STARTED) {
                launch {
                    viewModel.historyProducts.collect { products ->
                        productAdapter.submitList(products)
                        if (currentTabPosition == 0) {
                            updateEmptyState(products.isEmpty(), "No cached products in history yet")
                        }
                    }
                }
                launch {
                    viewModel.scanEvents.collect { events ->
                        scanEventAdapter.submitList(events)
                        if (currentTabPosition == 1) {
                            updateEmptyState(events.isEmpty(), "No scan events recorded yet")
                        }
                    }
                }
            }
        }
    }

    private fun updateEmptyState(isEmpty: Boolean, message: String) {
        if (isEmpty) {
            binding.tvEmptyHistory.visibility = View.VISIBLE
            binding.tvEmptyHistory.text = message
            binding.rvHistory.visibility = View.GONE
        } else {
            binding.tvEmptyHistory.visibility = View.GONE
            binding.rvHistory.visibility = View.VISIBLE
        }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}
