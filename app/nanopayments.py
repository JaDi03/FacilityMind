import os
import time
import uuid
import base64
import logging
import requests
import secrets
from dotenv import load_dotenv
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.serialization import load_pem_public_key
from cryptography.hazmat.backends import default_backend
from web3 import Web3

load_dotenv()
logger = logging.getLogger(__name__)

# ==============================================================================
# Circle Gateway x402 Real Developer-Controlled Wallet Manager (ARC-TESTNET)
# ==============================================================================

CIRCLE_API_KEY = os.getenv("CIRCLE_API_KEY", "")
CIRCLE_ENTITY_SECRET = os.getenv("CIRCLE_ENTITY_SECRET", "")  # 32-byte hex string
CIRCLE_WALLET_SET_ID = os.getenv("CIRCLE_WALLET_SET_ID", "")  # Container Wallet Set ID
CIRCLE_USDC_TOKEN_ID = os.getenv("CIRCLE_USDC_TOKEN_ID", "15dc2b5d-0994-58b0-bf8c-3a0501148ee8")  # Arc Testnet USDC Circle ID
CIRCLE_DESTINATION_ADDRESS = os.getenv("CIRCLE_DESTINATION_ADDRESS", "0x5cCA6233A071314c60EF2243D2CF68802aAF6F19")  # Target address to collect profits
CIRCLE_SOURCE_WALLET_ID = os.getenv("CIRCLE_SOURCE_WALLET_ID", "")  # Fallback buyer wallet ID

CIRCLE_BASE_URL = "https://api.circle.com/v1/w3s"
BLOCKCHAIN = "ARC-TESTNET" # Target Blockchain network

# Contract Addresses for Circle Gateway batched settlement on Arc Testnet
GATEWAY_CONTRACT = "0x0077777d7EBA4688BDeF3E311b846F25870A19B9"
USDC_CONTRACT = "0x3600000000000000000000000000000000000000"

# Gemini 2.5 Pro Base Pricing
INPUT_TOKEN_PRICE = 0.00000125
OUTPUT_TOKEN_PRICE = 0.000005
MARGIN_MULTIPLIER = 2.0

class X402NanopaymentGateway:
    """
    REAL developer-controlled programmable wallets manager (Circle Web3 Services).
    Allows client wallet generation, onchain balance inspection, Gateway contract deposit,
    and real gas-free EIP-3009 Circle Gateway batched settlement.
    """
    
    @staticmethod
    def _encrypt_entity_secret() -> str:
        """Fetch Circle's entity public key and encrypt the entity secret in memory using RSA-OAEP."""
        if not CIRCLE_ENTITY_SECRET:
            raise ValueError("Missing CIRCLE_ENTITY_SECRET in your .env file")

        headers = {
            "Authorization": f"Bearer {CIRCLE_API_KEY}",
            "Content-Type": "application/json"
        }
        res = requests.get(f"{CIRCLE_BASE_URL}/config/entity/publicKey", headers=headers)
        if res.status_code != 200:
            raise Exception(f"Failed to fetch Circle public key: {res.text}")
            
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
        Creates a new developer-controlled programmable wallet for a newly registered client.
        """
        if not CIRCLE_API_KEY or not CIRCLE_ENTITY_SECRET:
            raise ValueError("Circle master credentials (API_KEY or ENTITY_SECRET) are missing on the server.")

        headers = {
            "Authorization": f"Bearer {CIRCLE_API_KEY}",
            "Content-Type": "application/json"
        }

        # 1. Fetch or create the container Wallet Set
        wallet_set_id = CIRCLE_WALLET_SET_ID
        if not wallet_set_id:
            logger.info("[x402 REAL] Creating a new container Wallet Set...")
            ciphertext_set = X402NanopaymentGateway._encrypt_entity_secret()
            wallet_set_payload = {
                "idempotencyKey": str(uuid.uuid4()),
                "entitySecretCiphertext": ciphertext_set,
                "name": "FacilityMind Clients Wallet Set"
            }
            res = requests.post(f"{CIRCLE_BASE_URL}/developer/walletSets", json=wallet_set_payload, headers=headers)
            if res.status_code in [200, 201]:
                wallet_set_id = res.json().get("data", {}).get("walletSet", {}).get("id")
                logger.info(f"[x402 REAL] Wallet Set successfully created: {wallet_set_id}")
            else:
                raise Exception(f"Failed to create Wallet Set: {res.text}")

        # 2. Generate a real blockchain wallet with mandatory accountType parameter
        ciphertext_wallet = X402NanopaymentGateway._encrypt_entity_secret()
        wallet_payload = {
            "idempotencyKey": str(uuid.uuid4()),
            "entitySecretCiphertext": ciphertext_wallet,
            "blockchains": [BLOCKCHAIN],
            "count": 1,
            "walletSetId": wallet_set_id,
            "accountType": "EOA"  # Mandatory EOA parameter for developer-controlled wallets
        }
        
        logger.info(f"[x402 REAL] Generating blockchain wallet on {BLOCKCHAIN} for new client...")
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
        raise Exception(f"Failed to generate real wallet on Circle: {res.text}")

    @staticmethod
    def get_wallet_address(wallet_id: str) -> str:
        """
        Retrieves the blockchain address of a developer-controlled wallet ID.
        """
        if not CIRCLE_API_KEY or not wallet_id or wallet_id == "N/A":
            return ""

        headers = {
            "Authorization": f"Bearer {CIRCLE_API_KEY}",
            "Content-Type": "application/json"
        }
        
        try:
            res = requests.get(f"{CIRCLE_BASE_URL}/wallets/{wallet_id}", headers=headers)
            if res.status_code == 200:
                return res.json().get("data", {}).get("wallet", {}).get("address", "")
            return ""
        except Exception as e:
            logger.error(f"[x402 REAL] Wallet address query failed: {e}")
            return ""

    @staticmethod
    def get_wallet_balance(wallet_id: str) -> float:
        """
        Reads the real-time USDC onchain token balance of the client's wallet.
        """
        if not CIRCLE_API_KEY or not wallet_id or wallet_id == "N/A":
            return 0.0

        headers = {
            "Authorization": f"Bearer {CIRCLE_API_KEY}",
            "Content-Type": "application/json"
        }
        
        try:
            res = requests.get(f"{CIRCLE_BASE_URL}/wallets/{wallet_id}/balances", headers=headers)
            if res.status_code == 200:
                balances = res.json().get("data", {}).get("tokenBalances", [])
                for bal in balances:
                    token = bal.get("token", {})
                    if token.get("symbol") == "USDC":
                        return float(bal.get("amount", "0.0"))
            return 0.0
        except Exception as e:
            logger.error(f"[x402 REAL] Balance query failed: {e}")
            return 0.0

    @staticmethod
    def get_gateway_balance(wallet_address: str) -> float:
        """
        Reads the real-time USDC balance deposited inside the Gateway Contract for this depositor.
        """
        if not CIRCLE_API_KEY or not wallet_address:
            return 0.0

        headers = {
            "Authorization": f"Bearer {CIRCLE_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "token": "USDC",
            "sources": [
                {
                    "depositor": wallet_address,
                    "domain": 26  # ARC-TESTNET domain
                }
            ]
        }
        
        try:
            res = requests.post("https://gateway-api-testnet.circle.com/v1/balances", json=payload, headers=headers)
            if res.status_code == 200:
                balances = res.json().get("balances", [])
                if balances:
                    return float(balances[0].get("balance", "0.0"))
            return 0.0
        except Exception as e:
            logger.error(f"[x402 REAL] Gateway Balance query failed: {e}")
            return 0.0

    @staticmethod
    def deposit_to_gateway(wallet_id: str, amount_usdc: float) -> dict:
        """
        Deposits USDC into the Circle Gateway contract onchain on behalf of this wallet.
        Performs:
          1. ERC20 approve(GatewayWallet, amount)
          2. Gateway deposit(USDC, amount)
        """
        if not CIRCLE_API_KEY or not CIRCLE_ENTITY_SECRET or not wallet_id:
            raise ValueError("Circle API credentials or wallet ID are missing.")

        amount_atomic = str(int(amount_usdc * 1_000_000))
        headers = {
            "Authorization": f"Bearer {CIRCLE_API_KEY}",
            "Content-Type": "application/json"
        }

        # ─── STEP 1: APPROVE USDC SPEND ───
        logger.info(f"[x402 REAL] Initiating onchain USDC approval of {amount_usdc} USDC to Gateway...")
        ciphertext_approve = X402NanopaymentGateway._encrypt_entity_secret()
        approve_payload = {
            "idempotencyKey": str(uuid.uuid4()),
            "entitySecretCiphertext": ciphertext_approve,
            "walletId": wallet_id,
            "contractAddress": USDC_CONTRACT,
            "abiFunctionSignature": "approve(address,uint256)",
            "abiParameters": [GATEWAY_CONTRACT, amount_atomic],
            "feeLevel": "HIGH"
        }

        res_approve = requests.post(f"{CIRCLE_BASE_URL}/developer/transactions/contractExecution", json=approve_payload, headers=headers)
        if res_approve.status_code not in [200, 201]:
            raise Exception(f"USDC approval transaction failed: {res_approve.text}")
        
        approve_tx_id = res_approve.json().get("data", {}).get("id")
        logger.info(f"[x402 REAL] USDC approval initiated. Tx ID: {approve_tx_id}")

        # Wait for approval transaction to confirmed on blockchain (typically 10-15 seconds)
        logger.info("[x402 REAL] Waiting 15 seconds for approval block confirmation...")
        time.sleep(15)

        # ─── STEP 2: DEPOSIT TO GATEWAY CONTRACT ───
        logger.info(f"[x402 REAL] Initiating onchain Gateway contract deposit...")
        ciphertext_deposit = X402NanopaymentGateway._encrypt_entity_secret()
        deposit_payload = {
            "idempotencyKey": str(uuid.uuid4()),
            "entitySecretCiphertext": ciphertext_deposit,
            "walletId": wallet_id,
            "contractAddress": GATEWAY_CONTRACT,
            "abiFunctionSignature": "deposit(address,uint256)",
            "abiParameters": [USDC_CONTRACT, amount_atomic],
            "feeLevel": "HIGH"
        }

        res_deposit = requests.post(f"{CIRCLE_BASE_URL}/developer/transactions/contractExecution", json=deposit_payload, headers=headers)
        if res_deposit.status_code not in [200, 201]:
            raise Exception(f"Gateway deposit transaction failed: {res_deposit.text}")

        deposit_tx_id = res_deposit.json().get("data", {}).get("id")
        logger.info(f"[x402 REAL] Gateway deposit initiated! Tx ID: {deposit_tx_id}")

        return {
            "success": True,
            "approve_tx_id": approve_tx_id,
            "deposit_tx_id": deposit_tx_id
        }

    @staticmethod
    def process_payment(input_chars: int, output_chars: int, client_wallet_id: str = "") -> dict:
        """
        Processes real x402 gas-free nanopayments via Circle Gateway batched settlement.
        If the client's Gateway balance is 0, it falls back to direct on-chain billing,
        educating the client to deposit funds to the Gateway for gas-free sub-cent queries.
        """
        input_tokens = int(input_chars / 4)
        output_tokens = int(output_chars / 4)
        
        raw_cost = (input_tokens * INPUT_TOKEN_PRICE) + (output_tokens * OUTPUT_TOKEN_PRICE)
        final_charge = max(raw_cost * MARGIN_MULTIPLIER, 0.001)
        final_charge_rounded = round(final_charge, 4)
        value_atomic = int(final_charge_rounded * 1_000_000)

        source_wallet = client_wallet_id or CIRCLE_SOURCE_WALLET_ID
        
        if not all([CIRCLE_API_KEY, CIRCLE_ENTITY_SECRET, source_wallet, CIRCLE_DESTINATION_ADDRESS]):
            logger.error("[x402 REAL] Circle API credentials are incomplete on the server.")
            return {
                "network": "Incomplete Configuration",
                "currency": "USDC",
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "amount_charged": final_charge_rounded,
                "eip3009_auth": "ERROR: CHECK_YOUR_ENV",
                "tx_hash": "FAILED_MISSING_CREDENTIALS",
                "settlement_status": "FAILED_MISSING_ENV_KEYS"
            }

        # 1. Fetch wallet address
        wallet_address = X402NanopaymentGateway.get_wallet_address(source_wallet)
        if not wallet_address:
            logger.error(f"[x402 REAL] Could not locate wallet address for ID: {source_wallet}")
            return {
                "network": f"Circle Gateway ({BLOCKCHAIN})",
                "currency": "USDC",
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "amount_charged": final_charge_rounded,
                "eip3009_auth": "FAILED_WALLET_LOOKUP",
                "tx_hash": "FAILED_WALLET_LOOKUP",
                "settlement_status": "FAILED_WALLET_LOOKUP"
            }

        # 2. Check Gateway Balance
        gateway_balance = X402NanopaymentGateway.get_gateway_balance(wallet_address)
        logger.info(f"[x402 REAL] Client wallet {wallet_address} has {gateway_balance} USDC in Gateway contract.")

        # If Gateway balance is sufficient, perform real x402 Gasless Settle!
        if gateway_balance >= final_charge_rounded:
            try:
                logger.info(f"[x402 REAL] Processing gas-free Circle Gateway batch payment of {final_charge_rounded} USDC...")
                
                # A. Generate off-chain EIP-3009 signature using W3S sign/typedData API
                nonce_bytes = secrets.token_bytes(32)
                nonce_hex = "0x" + nonce_bytes.hex()
                
                now = int(time.time())
                valid_after = now - 600
                valid_before = now + (7 * 24 * 60 * 60) + 100
                
                typed_data = {
                    "types": {
                        "EIP712Domain": [
                            {"name": "name", "type": "string"},
                            {"name": "version", "type": "string"},
                            {"name": "chainId", "type": "uint256"},
                            {"name": "verifyingContract", "type": "address"}
                        ],
                        "TransferWithAuthorization": [
                            {"name": "from", "type": "address"},
                            {"name": "to", "type": "address"},
                            {"name": "value", "type": "uint256"},
                            {"name": "validAfter", "type": "uint256"},
                            {"name": "validBefore", "type": "uint256"},
                            {"name": "nonce", "type": "bytes32"}
                        ]
                    },
                    "primaryType": "TransferWithAuthorization",
                    "domain": {
                        "name": "GatewayWalletBatched",
                        "version": "1",
                        "chainId": 5042002,
                        "verifyingContract": Web3.to_checksum_address(GATEWAY_CONTRACT)
                    },
                    "message": {
                        "from": Web3.to_checksum_address(wallet_address),
                        "to": Web3.to_checksum_address(CIRCLE_DESTINATION_ADDRESS),
                        "value": str(value_atomic),
                        "validAfter": str(valid_after),
                        "validBefore": str(valid_before),
                        "nonce": nonce_hex
                    }
                }

                ciphertext = X402NanopaymentGateway._encrypt_entity_secret()
                sign_payload = {
                    "idempotencyKey": str(uuid.uuid4()),
                    "entitySecretCiphertext": ciphertext,
                    "walletId": source_wallet,
                    "data": base64.b64encode(json.dumps(typed_data).encode('utf-8')).decode('utf-8') if False else json.dumps(typed_data)
                }
                w3s_headers = {
                    "Authorization": f"Bearer {CIRCLE_API_KEY}",
                    "Content-Type": "application/json"
                }

                res_sign = requests.post("https://api.circle.com/v1/w3s/developer/sign/typedData", json=sign_payload, headers=w3s_headers)
                if res_sign.status_code != 200:
                    raise Exception(f"EIP-3009 signing failed: {res_sign.text}")

                signature = res_sign.json().get("data", {}).get("signature")
                
                # B. Submit the EIP-3009 payment payload to Circle Gateway Settle endpoint
                payment_requirements = {
                    "scheme": "exact",
                    "network": "eip155:5042002",
                    "asset": USDC_CONTRACT,
                    "amount": str(value_atomic),
                    "payTo": Web3.to_checksum_address(CIRCLE_DESTINATION_ADDRESS),
                    "maxTimeoutSeconds": 604900,
                    "extra": {
                        "name": "GatewayWalletBatched",
                        "version": "1",
                        "verifyingContract": Web3.to_checksum_address(GATEWAY_CONTRACT)
                    }
                }

                payment_payload = {
                    "x402Version": 2,
                    "resource": {
                        "url": "/api/chat",
                        "description": "FacilityMind AI blueprint query",
                        "mimeType": "application/json"
                    },
                    "payload": {
                        "authorization": {
                            "from": Web3.to_checksum_address(wallet_address),
                            "to": Web3.to_checksum_address(CIRCLE_DESTINATION_ADDRESS),
                            "value": str(value_atomic),
                            "validAfter": str(valid_after),
                            "validBefore": str(valid_before),
                            "nonce": nonce_hex
                        },
                        "signature": signature
                    },
                    "accepted": payment_requirements
                }

                settle_payload = {
                    "paymentPayload": payment_payload,
                    "paymentRequirements": payment_requirements
                }

                settle_headers = {
                    "Authorization": f"Bearer {CIRCLE_API_KEY}",
                    "Content-Type": "application/json"
                }

                res_settle = requests.post("https://gateway-api-testnet.circle.com/v1/x402/settle", json=settle_payload, headers=settle_headers)
                
                if res_settle.status_code == 200:
                    settle_res = res_settle.json()
                    if settle_res.get("success"):
                        tx_id = settle_res.get("transaction", "BATCHED")
                        logger.info(f"[x402 REAL] 🎉 Gasless Nanopayment settled successfully! Tx ID: {tx_id}")
                        return {
                            "network": f"Circle Gateway (x402 - {BLOCKCHAIN})",
                            "currency": "USDC",
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                            "amount_charged": final_charge_rounded,
                            "eip3009_auth": "GASLESS_EIP3009_SUCCESS",
                            "tx_hash": tx_id,
                            "settlement_status": "GASLESS_GATEWAY_SETTLED"
                        }
                    else:
                        raise Exception(f"Gateway reject: {settle_res.get('errorReason', 'unknown')}")
                else:
                    raise Exception(f"Gateway Settle API failed ({res_settle.status_code}): {res_settle.text}")

            except Exception as e:
                logger.error(f"[x402 REAL] Gasless settlement failed: {e}. Falling back to direct onchain transfer.")

        # Fallback to direct onchain token transfer (since Gateway balance is 0 or error occurred)
        logger.info("[x402 REAL] Gateway balance is insufficient or offline. Charging direct EOA balance onchain...")
        try:
            ciphertext = X402NanopaymentGateway._encrypt_entity_secret()
            headers = {
                "Authorization": f"Bearer {CIRCLE_API_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "idempotencyKey": str(uuid.uuid4()),
                "entitySecretCiphertext": ciphertext,
                "amounts": [str(final_charge_rounded)],
                "destinationAddress": CIRCLE_DESTINATION_ADDRESS,
                "feeLevel": "MEDIUM",
                "tokenId": CIRCLE_USDC_TOKEN_ID,
                "walletId": source_wallet
            }
            
            res = requests.post(f"{CIRCLE_BASE_URL}/developer/transactions/transfer", json=payload, headers=headers)
            
            if res.status_code in [200, 201]:
                tx_data = res.json().get("data", {})
                tx_id = tx_data.get("id", "TX_PENDING")
                logger.info(f"[x402 REAL] Direct onchain transfer dispatched! ID: {tx_id}")
                return {
                    "network": f"Onchain Transfer (Fallback - {BLOCKCHAIN})",
                    "currency": "USDC",
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "amount_charged": final_charge_rounded,
                    "eip3009_auth": "FALLBACK_ONCHAIN_GAS_CHARGED",
                    "tx_hash": tx_id,
                    "settlement_status": "SUCCESSFUL (Warning: Paid onchain gas fees! Deposit to Gateway contract in sidebar to unlock gasless payments)"
                }
            else:
                raise Exception(f"API Error: {res.text}")
                
        except Exception as e:
            logger.error(f"[x402 REAL] Onchain transfer fallback failed: {e}")
            return {
                "network": f"Circle Gateway ({BLOCKCHAIN})",
                "currency": "USDC",
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "amount_charged": final_charge_rounded,
                "eip3009_auth": "FAILED",
                "tx_hash": "BLOCKCHAIN_FAILURE",
                "settlement_status": f"FAILED: {str(e)[:50]}"
            }
