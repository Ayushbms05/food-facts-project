package com.example.foodfacts.data.local

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import kotlinx.coroutines.flow.Flow

@Dao
interface ScanEventDao {

    @Query("SELECT * FROM scan_events ORDER BY timestamp DESC")
    fun getAllEventsFlow(): Flow<List<ScanEventEntity>>

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertEvent(event: ScanEventEntity): Long

    @Query("DELETE FROM scan_events WHERE id = :id")
    suspend fun deleteById(id: Long)

    @Query("DELETE FROM scan_events")
    suspend fun clearAll()
}
