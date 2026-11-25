package com.example.foodfacts.ui.adapters

import android.view.LayoutInflater
import android.view.ViewGroup
import androidx.recyclerview.widget.DiffUtil
import androidx.recyclerview.widget.ListAdapter
import androidx.recyclerview.widget.RecyclerView
import com.example.foodfacts.data.model.ScanEvent
import com.example.foodfacts.data.model.ScanSource
import com.example.foodfacts.databinding.ItemScanEventBinding
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class ScanEventAdapter(
    private val onItemClick: (ScanEvent) -> Unit
) : ListAdapter<ScanEvent, ScanEventAdapter.ScanEventViewHolder>(DiffCallback) {

    private val dateFormat = SimpleDateFormat("MMM dd, yyyy HH:mm:ss", Locale.getDefault())

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ScanEventViewHolder {
        val binding = ItemScanEventBinding.inflate(
            LayoutInflater.from(parent.context), parent, false
        )
        return ScanEventViewHolder(binding)
    }

    override fun onBindViewHolder(holder: ScanEventViewHolder, position: Int) {
        holder.bind(getItem(position))
    }

    inner class ScanEventViewHolder(
        private val binding: ItemScanEventBinding
    ) : RecyclerView.ViewHolder(binding.root) {

        fun bind(event: ScanEvent) {
            binding.tvEventTitle.text = event.productName
            binding.tvEventSubtitle.text = "${event.brand} • GTIN-14: ${event.normalizedBarcode}"
            binding.tvEventStatus.text = "Status: ${event.lookupStatus} | NOVA: ${event.novaGroup ?: "N/A"}"
            binding.tvEventTimestamp.text = dateFormat.format(Date(event.timestamp))

            binding.tvSourceTag.text = when (event.scanSource) {
                ScanSource.BARCODE -> "BARCODE"
                ScanSource.OCR -> "OCR"
                ScanSource.MANUAL_SEARCH -> "SEARCH"
            }

            binding.root.setOnClickListener { onItemClick(event) }
        }
    }

    object DiffCallback : DiffUtil.ItemCallback<ScanEvent>() {
        override fun areItemsTheSame(oldItem: ScanEvent, newItem: ScanEvent): Boolean {
            return oldItem.id == newItem.id
        }

        override fun areContentsTheSame(oldItem: ScanEvent, newItem: ScanEvent): Boolean {
            return oldItem == newItem
        }
    }
}
