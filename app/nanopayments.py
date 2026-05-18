import os
import time
import uuid
import base64
import logging
import requests
from dotenv import load_dotenv
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.serialization import load_pem_public_key
from cryptography.hazmat.backends import default_backend

load_dotenv()
logger = logging.getLogger(__name__)

# ==============================================================================
# Circle Nanopayments REAL Web3 Gateway (Programmable Wallets)
# ==============================================================================

# Credentials (Deben estar en tu archivo .env)
CIRCLE_API_KEY = os.getenv("CIRCLE_API_KEY", "")
CIRCLE_ENTITY_SECRET = os.getenv("CIRCLE_ENTITY_SECRET", "")  # Hex de 32 bytes
CIRCLE_SOURCE_WALLET_ID = os.getenv("CIRCLE_SOURCE_WALLET_ID", "")
CIRCLE_USDC_TOKEN_ID = os.getenv("CIRCLE_USDC_TOKEN_ID", "")  # Token ID de USDC en Testnet
CIRCLE_DESTINATION_ADDRESS = os.getenv("CIRCLE_DESTINATION_ADDRESS", "")

CIRCLE_BASE_URL = "https://api.circle.com/v1/w3s"

# Gemini 2.5 Pro Base Pricing
INPUT_TOKEN_PRICE = 0.00000125  # $1.25 per 1M tokens
OUTPUT_TOKEN_PRICE = 0.000005   # $5.00 per 1M tokens
MARGIN_MULTIPLIER = 2.0         # 100% markup

class NanopaymentGateway:
    """
    Middleware REAL que intercepta el uso de tokens y ejecuta transacciones 
    criptográficas verdaderas en Circle usando Developer-Controlled Wallets.
    """
    
    @staticmethod
    def _get_entity_secret_ciphertext() -> str:
        """Fetch public key and encrypt the entity secret."""
        if not CIRCLE_ENTITY_SECRET:
            raise ValueError("Falta CIRCLE_ENTITY_SECRET en .env")

        headers = {
            "Authorization": f"Bearer {CIRCLE_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # 1. Fetch Entity Public Key
        res = requests.get(f"{CIRCLE_BASE_URL}/config/entity/publicKey", headers=headers)
        if res.status_code != 200:
            raise Exception(f"Error fetching Circle Public Key: {res.text}")
            
        data = res.json().get("data", {})
        pub_key_pem = data.get("publicKey")
        if not pub_key_pem:
            raise Exception("No public key returned by Circle")
            
        # 2. Encrypt Entity Secret
        public_key = load_pem_public_key(pub_key_pem.encode('utf-8'), default_backend())
        secret_bytes = bytes.fromhex(CIRCLE_ENTITY_SECRET)
        
        ciphertext = public_key.encrypt(
            secret_bytes,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return base64.b64encode(ciphertext).decode('utf-8')

    @staticmethod
    def process_payment(input_chars: int, output_chars: int) -> dict:
        input_tokens = int(input_chars / 4)
        output_tokens = int(output_chars / 4)
        
        raw_cost = (input_tokens * INPUT_TOKEN_PRICE) + (output_tokens * OUTPUT_TOKEN_PRICE)
        final_charge = max(raw_cost * MARGIN_MULTIPLIER, 0.005) # Minimo medio centavo
        
        if not all([CIRCLE_API_KEY, CIRCLE_ENTITY_SECRET, CIRCLE_SOURCE_WALLET_ID, CIRCLE_DESTINATION_ADDRESS]):
            logger.warning("[Nanopayments] Faltan credenciales reales en .env. Se registrará la orden pero no viajará a la blockchain.")
            return {
                "network": "Configuración Faltante",
                "currency": "USDC",
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "amount_charged": round(final_charge, 4),
                "tx_hash": "ERROR: REVISA_TU_ENV",
                "wallet": CIRCLE_SOURCE_WALLET_ID or "N/A",
                "status": "FAILED_MISSING_KEYS"
            }

        try:
            # Encriptar el secreto para autorizar la transacción
            ciphertext = NanopaymentGateway._get_entity_secret_ciphertext()
            
            # Ejecutar transacción real
            headers = {
                "Authorization": f"Bearer {CIRCLE_API_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "idempotencyKey": str(uuid.uuid4()),
                "entitySecretCiphertext": ciphertext,
                "amounts": [str(round(final_charge, 4))],
                "destinationAddress": CIRCLE_DESTINATION_ADDRESS,
                "feeLevel": "MEDIUM",
                "tokenId": CIRCLE_USDC_TOKEN_ID,
                "walletId": CIRCLE_SOURCE_WALLET_ID
            }
            
            res = requests.post(f"{CIRCLE_BASE_URL}/developer/transactions/transfer", json=payload, headers=headers)
            
            if res.status_code in [200, 201]:
                tx_data = res.json().get("data", {})
                tx_id = tx_data.get("id", "TX_PENDING")
                
                receipt = {
                    "network": "ARC/EVM Testnet",
                    "currency": "USDC",
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "amount_charged": round(final_charge, 4),
                    "tx_hash": tx_id,  # Este es el ID real de Circle
                    "wallet": CIRCLE_SOURCE_WALLET_ID,
                    "status": "PAID_AUTOMATICALLY"
                }
                logger.info(f"[Nanopayments REAL] Cobro exitoso: {final_charge} USDC. TX ID: {tx_id}")
                return receipt
            else:
                raise Exception(f"Fallo en Circle API: {res.text}")
                
        except Exception as e:
            logger.error(f"[Nanopayments REAL] Error: {str(e)}")
            return {
                "network": "ARC/EVM Testnet",
                "currency": "USDC",
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "amount_charged": round(final_charge, 4),
                "tx_hash": "FALLO_EN_RED",
                "wallet": CIRCLE_SOURCE_WALLET_ID,
                "status": f"FAILED: {str(e)[:50]}"
            }
