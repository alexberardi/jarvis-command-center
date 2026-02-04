"""Settings service with cascade lookup: Node > Household > Default.

This service manages scoped settings that can be defined at three levels:
- System default: household_id=NULL, node_id=NULL
- Household-wide: household_id set, node_id=NULL
- Node-specific: both household_id and node_id set

When looking up a setting, the service checks each level in order and returns
the most specific value found.
"""

from sqlalchemy.orm import Session

from app.models import Setting


class SettingsService:
    """Service for managing scoped settings."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def get_setting(
        self,
        key: str,
        household_id: str | None = None,
        node_id: str | None = None,
    ) -> str | None:
        """Get a setting value with cascade lookup.

        Lookup order: Node > Household > Default

        Args:
            key: The setting key to look up
            household_id: Optional household scope
            node_id: Optional node scope (requires household_id)

        Returns:
            The setting value, or None if not found at any level
        """
        # Level 1: Node-specific (requires both household_id and node_id)
        if household_id is not None and node_id is not None:
            setting = self.db.query(Setting).filter(
                Setting.key == key,
                Setting.household_id == household_id,
                Setting.node_id == node_id,
            ).first()
            if setting:
                return setting.value

        # Level 2: Household-wide (requires household_id, node_id must be NULL)
        if household_id is not None:
            setting = self.db.query(Setting).filter(
                Setting.key == key,
                Setting.household_id == household_id,
                Setting.node_id.is_(None),
            ).first()
            if setting:
                return setting.value

        # Level 3: System default (both NULL)
        setting = self.db.query(Setting).filter(
            Setting.key == key,
            Setting.household_id.is_(None),
            Setting.node_id.is_(None),
        ).first()
        if setting:
            return setting.value

        return None

    def set_setting(
        self,
        key: str,
        value: str,
        household_id: str | None = None,
        node_id: str | None = None,
    ) -> Setting:
        """Set a setting value at the specified scope.

        If a setting already exists at the exact scope, it will be updated.
        Otherwise, a new setting is created.

        Args:
            key: The setting key
            value: The setting value
            household_id: Optional household scope (None = system default)
            node_id: Optional node scope (None = household-wide)

        Returns:
            The created or updated Setting
        """
        # Build the filter for exact scope match
        # We need to handle NULL values specially since == None doesn't work
        query = self.db.query(Setting).filter(Setting.key == key)

        if household_id is None:
            query = query.filter(Setting.household_id.is_(None))
        else:
            query = query.filter(Setting.household_id == household_id)

        if node_id is None:
            query = query.filter(Setting.node_id.is_(None))
        else:
            query = query.filter(Setting.node_id == node_id)

        existing = query.first()

        if existing:
            existing.value = value
            self.db.commit()
            self.db.refresh(existing)
            return existing
        else:
            setting = Setting(
                key=key,
                value=value,
                household_id=household_id,
                node_id=node_id,
            )
            self.db.add(setting)
            self.db.commit()
            self.db.refresh(setting)
            return setting

    def get_all_settings(
        self,
        household_id: str | None = None,
        node_id: str | None = None,
    ) -> dict[str, str]:
        """Get all settings with cascade override.

        Returns a dictionary of all setting keys with their effective values
        for the given scope. Settings are merged with the following priority:
        Node > Household > Default

        Args:
            household_id: Optional household scope
            node_id: Optional node scope

        Returns:
            Dictionary mapping setting keys to their effective values
        """
        result: dict[str, str] = {}

        # Start with system defaults (lowest priority)
        defaults = self.db.query(Setting).filter(
            Setting.household_id.is_(None),
            Setting.node_id.is_(None),
        ).all()
        for setting in defaults:
            result[setting.key] = setting.value

        # Override with household settings
        if household_id is not None:
            household_settings = self.db.query(Setting).filter(
                Setting.household_id == household_id,
                Setting.node_id.is_(None),
            ).all()
            for setting in household_settings:
                result[setting.key] = setting.value

        # Override with node settings (highest priority)
        if household_id is not None and node_id is not None:
            node_settings = self.db.query(Setting).filter(
                Setting.household_id == household_id,
                Setting.node_id == node_id,
            ).all()
            for setting in node_settings:
                result[setting.key] = setting.value

        return result

    def delete_setting(
        self,
        key: str,
        household_id: str | None = None,
        node_id: str | None = None,
    ) -> bool:
        """Delete a setting at the specified scope.

        Args:
            key: The setting key to delete
            household_id: The household scope (None = system default)
            node_id: The node scope (None = household-wide)

        Returns:
            True if a setting was deleted, False if not found
        """
        # Build exact scope filter
        query = self.db.query(Setting).filter(Setting.key == key)

        if household_id is None:
            query = query.filter(Setting.household_id.is_(None))
        else:
            query = query.filter(Setting.household_id == household_id)

        if node_id is None:
            query = query.filter(Setting.node_id.is_(None))
        else:
            query = query.filter(Setting.node_id == node_id)

        setting = query.first()

        if setting:
            self.db.delete(setting)
            self.db.commit()
            return True

        return False
