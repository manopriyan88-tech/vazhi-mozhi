"""In-memory Kitchen State Store.

Manages dynamic 3rd-party API keys, endpoints, and OpenAPI connectors dropped
into the Adaptive API Kitchen at runtime.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from app.core.models import EndpointSpec, KitchenAPI

logger = logging.getLogger("app.services.kitchen_store")


class KitchenStore:
    """Manages registered API connectors in the Adaptive Kitchen."""

    def __init__(self):
        self._apis: Dict[str, KitchenAPI] = {}

    def add_api(self, api: KitchenAPI) -> None:
        """Registers or updates an API connector."""
        self._apis[api.api_name] = api
        logger.info(f"Registered API '{api.api_name}' ({api.base_url})")

    def get_api(self, api_name: str) -> Optional[KitchenAPI]:
        """Retrieves an API connector by name."""
        return self._apis.get(api_name)

    def get_all(self) -> List[KitchenAPI]:
        """Returns all registered API connectors."""
        return list(self._apis.values())

    def remove_api(self, api_name: str) -> bool:
        """Removes an API connector."""
        if api_name in self._apis:
            del self._apis[api_name]
            return True
        return False

    def clear(self) -> None:
        """Clears all registered APIs."""
        self._apis.clear()

    def seed_demo_apis(self) -> List[KitchenAPI]:
        """Pre-loads 3 free public API connectors + 1 mock custom API key slot.

        Connectors:
        1. Open-Meteo Weather API (free, no key required)
        2. GitHub Public REST API (free, no key required)
        3. CoinGecko Public Crypto API (free, no key required)
        4. Mock Custom API Key Slot (pre-configured with Bearer token authentication)
        """
        demo_connectors = [
            KitchenAPI(
                api_name="open_meteo_weather",
                base_url="https://api.open-meteo.com",
                description="Free open-source weather forecast API for global temperature and humidity.",
                auth_type="none",
                endpoints=[
                    EndpointSpec(
                        path="/v1/forecast",
                        method="GET",
                        summary="Fetch live weather metrics (temperature, humidity)",
                        parameters={
                            "latitude": 35.6895,
                            "longitude": 139.6917,
                            "current": "temperature_2m,relative_humidity_2m",
                        },
                    )
                ],
            ),
            KitchenAPI(
                api_name="github_public",
                base_url="https://api.github.com",
                description="GitHub public REST API for repository metrics, stars, and user profiles.",
                auth_type="none",
                auth_header_name="User-Agent",
                api_key="API-Kitchen-Demo/1.0",
                endpoints=[
                    EndpointSpec(
                        path="/repos/fastapi/fastapi",
                        method="GET",
                        summary="Get repository statistics for FastAPI",
                        parameters={},
                    ),
                    EndpointSpec(
                        path="/users/octocat",
                        method="GET",
                        summary="Get public GitHub profile for octocat",
                        parameters={},
                    ),
                ],
            ),
            KitchenAPI(
                api_name="coingecko_crypto",
                base_url="https://api.coingecko.com",
                description="Free CoinGecko public API for real-time cryptocurrency spot prices.",
                auth_type="none",
                auth_header_name="User-Agent",
                api_key="API-Kitchen-Demo/1.0",
                endpoints=[
                    EndpointSpec(
                        path="/api/v3/simple/price",
                        method="GET",
                        summary="Fetch current spot price in USD for Bitcoin and Ethereum",
                        parameters={"ids": "bitcoin,ethereum", "vs_currencies": "usd"},
                    )
                ],
            ),
            KitchenAPI(
                api_name="mock_custom_api_slot",
                base_url="https://jsonplaceholder.typicode.com",
                description="Mock slot configured for testing user-provided API keys with Bearer auth injection.",
                api_key="mock_sk_live_kitchen_token_demo_98765",
                auth_type="bearer",
                auth_header_name="Authorization",
                endpoints=[
                    EndpointSpec(
                        path="/posts/1",
                        method="GET",
                        summary="Retrieve test post with mock authentication token",
                        parameters={},
                    )
                ],
            ),
        ]

        for connector in demo_connectors:
            self.add_api(connector)

        return self.get_all()


# Global in-memory kitchen store
kitchen_store = KitchenStore()


def get_kitchen_store() -> KitchenStore:
    """Dependency / accessor for the global kitchen store."""
    return kitchen_store
