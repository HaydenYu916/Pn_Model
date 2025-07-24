from openelectricity import AsyncOEClient
from openelectricity.types import DataMetric
import asyncio
from datetime import datetime, timedelta, timezone
import pandas as pd

async def main():
    async with AsyncOEClient() as client:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=2)  # ⚠️ 限制时间范围，避免超大请求
        response = await client.get_network_data(
            network_code="NEM",
            metrics=[DataMetric.ENERGY],
            interval="1h",
            date_start=start,
            date_end=end,
        )

        df = response.datatable.to_pandas()
        print(df.head())
        df.to_csv("fueltech_energy.csv", index=False)

asyncio.run(main())
