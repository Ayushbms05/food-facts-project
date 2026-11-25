package com.example.foodfacts.ui.adapters

import android.view.LayoutInflater
import android.view.ViewGroup
import androidx.core.content.ContextCompat
import androidx.recyclerview.widget.DiffUtil
import androidx.recyclerview.widget.ListAdapter
import androidx.recyclerview.widget.RecyclerView
import com.example.foodfacts.data.model.ComparisonWinner
import com.example.foodfacts.data.model.MetricComparison
import com.example.foodfacts.databinding.ItemMetricComparisonBinding

class MetricAdapter : ListAdapter<MetricComparison, MetricAdapter.MetricViewHolder>(DiffCallback) {

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): MetricViewHolder {
        val binding = ItemMetricComparisonBinding.inflate(
            LayoutInflater.from(parent.context), parent, false
        )
        return MetricViewHolder(binding)
    }

    override fun onBindViewHolder(holder: MetricViewHolder, position: Int) {
        holder.bind(getItem(position))
    }

    inner class MetricViewHolder(
        private val binding: ItemMetricComparisonBinding
    ) : RecyclerView.ViewHolder(binding.root) {

        fun bind(metric: MetricComparison) {
            binding.tvMetricName.text = metric.metricName

            val valAStr = metric.valueA?.let { "%.1f %s".format(it, metric.unit) } ?: "N/A"
            val valBStr = metric.valueB?.let { "%.1f %s".format(it, metric.unit) } ?: "N/A"

            binding.tvValueA.text = valAStr
            binding.tvValueB.text = valBStr

            val green = ContextCompat.getColor(binding.root.context, android.R.color.holo_green_dark)
            val defaultColor = ContextCompat.getColor(binding.root.context, android.R.color.tab_indicator_text)

            when (metric.winner) {
                ComparisonWinner.PRODUCT_A -> {
                    binding.tvValueA.setTextColor(green)
                    binding.tvValueB.setTextColor(defaultColor)
                }
                ComparisonWinner.PRODUCT_B -> {
                    binding.tvValueA.setTextColor(defaultColor)
                    binding.tvValueB.setTextColor(green)
                }
                else -> {
                    binding.tvValueA.setTextColor(defaultColor)
                    binding.tvValueB.setTextColor(defaultColor)
                }
            }
        }
    }

    object DiffCallback : DiffUtil.ItemCallback<MetricComparison>() {
        override fun areItemsTheSame(oldItem: MetricComparison, newItem: MetricComparison): Boolean {
            return oldItem.metricName == newItem.metricName
        }

        override fun areContentsTheSame(oldItem: MetricComparison, newItem: MetricComparison): Boolean {
            return oldItem == newItem
        }
    }
}
