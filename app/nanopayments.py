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
# Circle Gateway x402 Real Developer-Controlled Wallet Manager
# ==============================================================================

CIRCLE_API_KEY = os.getenv("CIRCLE_API_KEY", "")
CIRCLE_ENTITY_SECRET = os.getenv("CIRCLE_ENTITY_SECRET", "")  # Secreto Hex de 32 bytes
CIRCLE_WALLET_SET_ID = os.getenv("CIRCLE_WALLET_SET_ID", "")  # Wallet Set ID contenedor
CIRCLE_USDC_TOKEN_ID = os.getenv("CIRCLE_USDC_TOKEN_ID", "")  # ID del token USDC en testnet
CIRCLE_DESTINATION_ADDRESS = os.getenv("CIRCLE_DESTINATION_ADDRESS", "")  # Wallet donde recibes tus ganancias de la IA

CIRCLE_BASE_URL = "https://api.circle.com/v1/w3s"
BLOCKCHAIN = "AVAX-FUJI" # Red por defecto para testnet rápida

# Gemini 2.5 Pro Base Pricing
INPUT_TOKEN_PRICE = 0.00000125
OUTPUT_TOKEN_PRICE = 0.000005
MARGIN_MULTIPLIER = 2.0

class X402NanopaymentGateway:
    """
    Controlador REAL de wallets controladas por desarrollador (Circle Programmable Wallets).
    Permite creación de wallets para clientes, consulta de saldos onchain y cobros reales.
    """
    
    @staticmethod
    def _encrypt_entity_secret() -> str:
        """Obtiene la llave publica de Circle y encripta el Entity Secret en memoria."""
        if not CIRCLE_ENTITY_SECRET:
            raise ValueError("Falta CIRCLE_ENTITY_SECRET en tu archivo .env")

        headers = {
            "Authorization": f"Bearer {CIRCLE_API_KEY}",
            "Content-Type": "application/json"
        }
        res = requests.get(f"{CIRCLE_BASE_URL}/config/entity/publicKey", headers=headers)
        if res.status_code != 200:
            raise Exception(f"Fallo al obtener la llave publica de Circle: {res.text}")
            
        pub_key_pem = res.json().get("data", {}).get("publicKey")
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
    def create_client_wallet() -> dict:
        """
        Crea de forma 100% REAL una nueva wallet programable controlada por el 
        desarrollador para el cliente que acaba de entrar a la plataforma.
        """
        if not CIRCLE_API_KEY or not CIRCLE_ENTITY_SECRET:
            raise ValueError("Faltan las credenciales maestras de Circle (API_KEY o ENTITY_SECRET) en el servidor.")

        ciphertext = X402NanopaymentGateway._encrypt_entity_secret()
        headers = {
            "Authorization": f"Bearer {CIRCLE_API_KEY}",
            "Content-Type": "application/json"
        }

        # 1. Obtener o Crear un Wallet Set para el proyecto si no está definido
        wallet_set_id = CIRCLE_WALLET_SET_ID
        if not wallet_set_id:
            logger.info("[x402 REAL] Creando nuevo Wallet Set contenedor...")
            wallet_set_payload = {
                "idempotencyKey": str(uuid.uuid4()),
                "entitySecretCiphertext": ciphertext,
                "name": "FacilityMind Clients Wallet Set"
            }
            res = requests.post(f"{CIRCLE_BASE_URL}/developer/walletSets", json=wallet_set_payload, headers=headers)
            if res.status_code in [200, 201]:
                wallet_set_id = res.json().get("data", {}).get("walletSet", {}).get("id")
                # Guardar temporalmente en memoria/config
                logger.info(f"[x402 REAL] Wallet Set creado con éxito: {wallet_set_id}")
            else:
                raise Exception(f"Fallo al crear Wallet Set: {res.text}")

        # 2. Generar la wallet real en la Blockchain
        wallet_payload = {
            "idempotencyKey": str(uuid.uuid4()),
            "entitySecretCiphertext": ciphertext,
            "blockchains": [BLOCKCHAIN],
            "count": 1,
            "walletSetId": wallet_set_id
        }
        
        logger.info(f"[x402 REAL] Generando wallet real en {BLOCKCHAIN} para nuevo cliente...")
        res = requests.post(f"{CIRCLE_BASE_URL}/developer/wallets", json=wallet_payload, headers=headers)
        
        if res.status_code in [200, 201]:
            wallets = res.json().get("data", {}).get("wallets", [])
            if wallets:
                wallet_data = wallets[0]
                return {
                    "wallet_id": wallet_data.get("id"),
                    "address": wallet_data.get("address"),
                    "blockchain": BLOCKCHAIN,
                    "success": True
                }
        raise Exception(f"Fallo al generar wallet real en Circle: {res.text}")

    @staticmethod
    def get_wallet_balance(wallet_id: str) -> float:
        """
        Consulta onchain en tiempo real el saldo de USDC de la wallet del cliente.
        """
        if not CIRCLE_API_KEY or not wallet_id or wallet_id == "N/A":
            return 0.0

        headers = {
            "Authorization": f"Bearer {CIRCLE_API_KEY}",
            "Content-Type": "application/json"
        }
        
        try:
            res = requests.get(f"{CIRCLE_BASE_URL}/developer/wallets/{wallet_id}/balances", headers=headers)
            if res.status_code == 200:
                balances = res.json().get("data", {}).get("tokenBalances", [])
                for bal in balances:
                    token = bal.get("token", {})
                    # Validar si es USDC
                    if token.get("symbol") == "USDC":
                        return float(bal.get("amount", "0.0"))
            return 0.0
        except Exception as e:
            logger.error(f"[x402 REAL] Error al consultar saldo: {e}")
            return 0.0

    @staticmethod
    def process_payment(input_chars: int, output_chars: int, client_wallet_id: str = "") -> dict:
        """
        Ejecuta la transaccion real de nanopago debitando saldo de la wallet del cliente.
        """
        input_tokens = int(input_chars / 4)
        output_tokens = int(output_chars / 4)
        
        raw_cost = (input_tokens * INPUT_TOKEN_PRICE) + (output_tokens * OUTPUT_TOKEN_PRICE)
        final_charge = max(raw_cost * MARGIN_MULTIPLIER, 0.005)
        
        # Si no se pasó una wallet específica de cliente, usar la global de pruebas
        source_wallet = client_wallet_id or CIRCLE_SOURCE_WALLET_ID
        
        if not all([CIRCLE_API_KEY, CIRCLE_ENTITY_SECRET, source_wallet, CIRCLE_DESTINATION_ADDRESS]):
            logger.error("[x402 REAL] Credenciales de Circle incompletas en el servidor.")
            return {
                "network": "Configuración Incompleta",
                "currency": "USDC",
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "amount_charged": round(final_charge, 4),
                "eip3009_auth": "ERROR: REVISA_TU_ENV",
                "tx_hash": "FALLO_SIN_CREDENCIALES",
                "settlement_status": "FAILED_MISSING_ENV_KEYS"
            }

        try:
            ciphertext = X402NanopaymentGateway._encrypt_entity_secret()
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
                "walletId": source_wallet
            }
            
            res = requests.post(f"{CIRCLE_BASE_URL}/developer/transactions/transfer", json=payload, headers=headers)
            
            if res.status_code in [200, 201]:
                tx_data = res.json().get("data", {})
                tx_id = tx_data.get("id", "TX_PENDING")
                
                receipt = {
                    "network": f"Circle Gateway (x402 - {BLOCKCHAIN})",
                    "currency": "USDC",
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "amount_charged": round(final_charge, 4),
                    "eip3009_auth": "EIP3009_Sponsor_Success",
                    "tx_hash": tx_id,
                    "settlement_status": "REAL_TRANSACTION_BROADCASTED"
                }
                logger.info(f"[x402 REAL] 🎉 Transacción blockchain completada! ID: {tx_id}")
                return receipt
            else:
                raise Exception(f"API Error: {res.text}")
                
        except Exception as e:
            logger.error(f"[x402 REAL] Fallo al procesar nanopago: {str(e)}")
            return {
                "network": f"Circle Gateway ({BLOCKCHAIN})",
                "currency": "USDC",
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "amount_charged": round(final_charge, 4),
                "eip3009_auth": "FAILED",
                "tx_hash": "FALLO_BLOCKCHAIN",
                "settlement_status": f"FAILED: {str(e)[:50]}"
            }
